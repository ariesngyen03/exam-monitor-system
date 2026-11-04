"""
main.py — AI Server của ExamMonitorSystem.

Nhận FramePacket (JPEG) từ C++ Client qua ZeroMQ, chạy ExamAIService (YOLO người -> sơ đồ chỗ ngồi ->
Face Mesh -> đặc trưng 13 chiều -> luật hành vi + Transformer -> cảm xúc), ghi log/bằng chứng theo phiên,
rồi trả InferenceResult về C++.

    C++ ZmqPublisher  (PUB bind  tcp://*:5555) ──FramePacket──────▶ SUB connect (server này)
    C++ ZmqSubscriber (SUB conn. tcp://127.0.0.1:5556) ◀──InferenceResult── PUB bind tcp://*:5556

Phiên giám sát: bắt đầu khi nhận frame đầu tiên, kết thúc khi C++ ngừng gửi frame quá --idle-timeout giây
(bấm DỪNG LẠI) hoặc khi tắt server (Ctrl+C). Mỗi phiên mới đều học lại sơ đồ chỗ ngồi.

Chạy (Python 3.10/3.11 — mediapipe.solutions không có trên 3.12+):
    python python_ai/main.py --room PHONG_THI_01 --proctor "Nguyen Van A"
"""
import os

# Phải đặt TRƯỚC khi import protobuf/mediapipe (giống ai_core.py)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import argparse
import sys
import threading
import time
import types

import cv2
import numpy as np
import zmq

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from ai_core import Config  # noqa: E402  (nạp sau khi đặt biến môi trường)
from exam_ai_service import ExamAIService  # noqa: E402
from session import ExamSession  # noqa: E402


def load_proto():
    """Nạp exam_monitor_pb2 từ proto/python_out.

    File pb2 sinh bởi protoc 28.x gọi google.protobuf.runtime_version (chỉ có từ protobuf 5.26), còn
    mediapipe 0.10.x bắt buộc protobuf 4.x. Định dạng nhị phân giống hệt nhau nên chỉ cần bỏ qua bước
    kiểm tra phiên bản khi runtime cũ hơn.
    """
    try:
        from google.protobuf import runtime_version  # noqa: F401
    except ImportError:
        import google.protobuf
        shim = types.ModuleType("google.protobuf.runtime_version")
        shim.Domain = types.SimpleNamespace(PUBLIC=0)
        shim.ValidateProtobufRuntimeVersion = lambda *args, **kwargs: None
        sys.modules["google.protobuf.runtime_version"] = shim
        google.protobuf.runtime_version = shim
    sys.path.insert(0, os.path.join(ROOT_DIR, "proto", "python_out"))
    import exam_monitor_pb2
    return exam_monitor_pb2


pb = load_proto()


def parse_args():
    p = argparse.ArgumentParser(description="ExamMonitorSystem — AI Server (ZeroMQ + Protobuf)")
    p.add_argument("--room", default=os.environ.get("EXAM_ROOM", "PHONG_THI_01"), help="Mã phòng thi")
    p.add_argument("--proctor", default=os.environ.get("EXAM_PROCTOR", ""),
                   help="Tên giám thị (để trống = tên tài khoản Windows)")
    p.add_argument("--fraud-threshold", type=float, default=70.0,
                   help="%% rủi ro tối thiểu để chụp thêm bằng chứng khi sự cố kéo dài")
    p.add_argument("--seatmap", default=None, help="File JSON từ seat_map_editor.py — dùng sơ đồ cố định, bỏ tự học")
    p.add_argument("--emotion", action="store_true",
                   help="BẬT nhánh cảm xúc (DeepFace). Mặc định tắt: suy luận cảm xúc từ khuôn mặt thiếu cơ sở khoa "
                        "học để kết luận gian lận và là dữ liệu nhạy cảm (NĐ 13/2023)")
    p.add_argument("--no-emotion", action="store_true", help=argparse.SUPPRESS)   # tương thích lệnh cũ
    p.add_argument("--sensitivity", choices=["low", "medium", "high"], default=os.environ.get("EXAM_SENSITIVITY", "medium"),
                   help="Độ nhạy SeatScorer: low = ít báo sai nhất, high = bắt nhiều nhất (xem training/train_seat_scorer.py)")
    p.add_argument("--sub", default="tcp://127.0.0.1:5555", help="Địa chỉ nhận frame từ C++")
    p.add_argument("--pub", default="tcp://*:5556", help="Địa chỉ trả kết quả về C++")
    p.add_argument("--idle-timeout", type=float, default=5.0,
                   help="Số giây không nhận frame thì coi như kết thúc phiên")
    p.add_argument("--data-dir", default=os.environ.get("EXAM_DATA_DIR") or os.path.join(ROOT_DIR, "data"),
                   help="Thư mục lưu evidence/, reports/, logs/")
    p.add_argument("--stop-file", default=None,
                   help="Khi file này xuất hiện: đóng phiên (ghi báo cáo) rồi thoát — dùng bởi Start.bat bản đóng gói")
    p.add_argument("--frame-width", type=int, default=1280)
    p.add_argument("--frame-height", type=int, default=720)
    return p.parse_args()


def configure(args):
    data_dir = os.path.abspath(args.data_dir)
    Config.EVIDENCE_DIR = os.path.join(data_dir, "evidence")
    Config.JSON_REPORT_DIR = os.path.join(data_dir, "reports")
    Config.JSON_REPORT_FILE = os.path.join(data_dir, "reports", "session_reports.json")
    Config.LOG_FILE = os.path.join(data_dir, "logs", "cheating_log.csv")
    Config.PERSON_YOLO_MODEL = os.path.join(BASE_DIR, "models", "yolov8s.pt")
    Config.EXAM_ROOM_ID = args.room
    Config.PROCTOR_NAME = args.proctor
    Config.FRAUD_CAPTURE_THRESHOLD = args.fraud_threshold
    Config.FRAME_WIDTH = args.frame_width
    Config.FRAME_HEIGHT = args.frame_height
    Config.SEAT_SCORER_SENSITIVITY = args.sensitivity
    for d in (Config.EVIDENCE_DIR, Config.JSON_REPORT_DIR, os.path.dirname(Config.LOG_FILE)):
        os.makedirs(d, exist_ok=True)
    return os.path.dirname(Config.LOG_FILE)


def build_result(res, frame_id, alerts, session_id):
    out = pb.InferenceResult()
    out.timestamp = int(time.time() * 1000)
    out.frame_id = frame_id
    out.is_heartbeat = False
    out.calibrating = bool(res["calibrating"])
    out.calib_progress = float(res["progress"])
    out.n_people = int(res["n_people"])
    out.n_faces = int(res["n_faces"])
    out.session_id = session_id or ""
    for d in res["detections"]:
        det = out.detections.add()
        det.track_id = int(d["track_id"])
        det.behavior_type = d["behavior_type"]
        det.confidence = float(d["confidence"])
        det.bbox.extend(float(b) for b in d["bbox"])
        det.desk_id = int(d["desk_id"])
        det.seat_label = d["seat_label"] or ""
        det.violations.extend(str(v) for v in d["violations"])
        det.emotion = d["emotion"] or ""
        det.face_bbox.extend(float(b) for b in d.get("face_bbox") or ())
        det.is_present = bool(d["is_present"])
        det.is_cheating = bool(d["is_cheating"])
        det.risk_level = d["risk_level"] or ""
        alert = alerts.get(d["seat_idx"]) if d["seat_idx"] is not None else None
        if alert:
            det.new_alert = True
            det.evidence_path, det.new_incident = alert
    return out


def lower_priority():
    """AI chỉ cần 2 nhịp/giây: chạy ưu tiên thấp hơn bình thường để camera + giao diện C++ (cùng máy) luôn được CPU
    trước — trước đây YOLO chiếm 4 luồng CPU ~200 ms mỗi nhịp, giành CPU với luồng đọc camera và vẽ giao diện."""
    if os.name == "nt":
        import ctypes
        BELOW_NORMAL_PRIORITY_CLASS = 0x4000
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p      # HANDLE 64-bit: mặc định int sẽ cắt mất handle
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        if k32.SetPriorityClass(k32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS):
            print("[AI SERVER] Chạy ở mức ưu tiên thấp (BELOW_NORMAL) để không làm giật camera/giao diện.")


def heartbeat():
    hb = pb.InferenceResult()
    hb.timestamp = int(time.time() * 1000)
    hb.is_heartbeat = True
    return hb.SerializeToString()


class AIServer:
    def __init__(self, args):
        self.args = args
        self.log_dir = configure(args)
        self.service = ExamAIService(Config, seatmap_path=args.seatmap,
                                     enable_emotion=args.emotion and not args.no_emotion)
        if not self.service.detector.ready:
            raise RuntimeError("Không nạp được detector người (YOLO). Cài: pip install ultralytics")
        self.fixed_seatmap = not self.service.seat_map.calibrating
        self.session = None
        self.t0_ms = 0
        self.reset_seq = 0
        self.period = 1.0 / max(Config.FEATURE_HZ, 0.1)   # Transformer được huấn luyện ở đúng nhịp này
        self._err_ts = 0.0
        self._proc_ms, self._proc_report_ts = [], time.time()

    def warmup(self):
        """Lượt suy luận đầu của YOLO và FaceMesh mất vài giây — chạy trước để C++ không thấy AI 'treo' lúc bắt đầu."""
        t = time.time()
        self.service.detector.detect(np.zeros((Config.FRAME_HEIGHT, Config.FRAME_WIDTH, 3), np.uint8))
        if self.service.faces is not None and getattr(self.service.faces, "mesh", None) is not None:
            self.service.faces.mesh.process(np.zeros((256, 256, 3), np.uint8))
        print(f"[AI SERVER] Warm-up YOLO + FaceMesh xong sau {time.time() - t:.1f}s.")

    def relearn_seats(self):
        if self.fixed_seatmap:
            self.service.pipeline.engine._st.clear()
        else:
            self.service.reset_seatmap()
            print(f"[AI SERVER] Đang học sơ đồ chỗ ngồi trong {Config.SEAT_CALIBRATION_SECONDS:.0f}s "
                  f"(thí sinh nên ngồi yên)...")

    def start_session(self, packet):
        # Phòng/giám thị từ trang Cài đặt của C++ được ưu tiên hơn tham số dòng lệnh
        room = packet.exam_room or self.args.room
        proctor = packet.proctor or self.args.proctor
        Config.EXAM_ROOM_ID, Config.PROCTOR_NAME = room, proctor
        self.relearn_seats()
        self.session = ExamSession(room, proctor, self.log_dir)
        self.t0_ms = packet.timestamp
        self.reset_seq = packet.reset_seq

    def end_session(self):
        if self.session is not None:
            self.session.close()
            self.session = None

    def process(self, raw_msg):
        packet = pb.FramePacket()
        packet.ParseFromString(raw_msg)
        frame = cv2.imdecode(np.frombuffer(packet.image_data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return None
        size = (frame.shape[1], frame.shape[0])
        if size != getattr(self, "_frame_size", None):   # ghi vào ai_console.log để đối chiếu máy chạy thử
            self._frame_size = size
            print(f"[AI SERVER] Nhận khung hình {size[0]}x{size[1]} ({len(packet.image_data) // 1024} KB JPEG).", flush=True)
        if self.session is None:
            self.service.update_runtime_config(
                packet.sensitivity, packet.absent_alert_seconds, packet.seatmap_path
            )
            self.fixed_seatmap = not self.service.seat_map.calibrating
            self.start_session(packet)
        else:
            seatmap_changed = self.service.update_runtime_config(
                packet.sensitivity, packet.absent_alert_seconds, packet.seatmap_path
            )
            self.fixed_seatmap = not self.service.seat_map.calibrating
            if seatmap_changed:
                self.session.reset_tracking()
            if packet.reset_seq != self.reset_seq:
                self.reset_seq = packet.reset_seq
                print("[AI SERVER] Giao diện yêu cầu học lại sơ đồ chỗ ngồi.")
                self.session.reset_tracking()
                self.relearn_seats()

        t_sec = max(0.0, (packet.timestamp - self.t0_ms) / 1000.0)
        res = self.service.process_frame(frame, t_sec)

        alerts = {}
        if res["relearned"]:
            self.session.reset_tracking()
        elif not res["calibrating"]:
            if res["just_locked"]:
                print(f"[AI SERVER] Đã học xong {len(self.service.seat_map.seats)} chỗ ngồi — bắt đầu giám sát.")
            alerts = self.session.handle(res["snapshot"], frame, self.service.pipeline.seat_map)
            for idx, (path, is_new) in alerts.items():
                kind = "SỰ CỐ MỚI" if is_new else "bằng chứng thêm"
                print(f"[ALERT] Chỗ {self.service.seat_map.label(idx)} ({kind}) -> {os.path.basename(path)}")
        return build_result(res, packet.frame_id, alerts, self.session.session_id).SerializeToString()

    def run(self):
        self.warmup()
        ctx = zmq.Context()
        sub = ctx.socket(zmq.SUB)
        sub.setsockopt(zmq.CONFLATE, 1)          # chỉ giữ frame mới nhất (chống trễ)
        sub.setsockopt(zmq.RCVTIMEO, 25)
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        sub.connect(self.args.sub)
        pub = ctx.socket(zmq.PUB)
        pub.setsockopt(zmq.LINGER, 0)
        pub.bind(self.args.pub)
        print(f"[AI SERVER] Sẵn sàng: nhận frame {self.args.sub} | trả kết quả {self.args.pub} | "
              f"phân tích {Config.FEATURE_HZ:.0f} nhịp/giây. Ctrl+C để dừng.")

        pub_lock = threading.Lock()
        stop_hb = threading.Event()
        last_send = [time.time()]

        def _hb_loop():
            while not stop_hb.wait(1.0):
                if time.time() - last_send[0] >= 0.9:
                    with pub_lock:
                        try:
                            pub.send(heartbeat())
                            last_send[0] = time.time()
                        except Exception:
                            pass

        hb_thread = threading.Thread(target=_hb_loop, daemon=True, name="HeartbeatThread")
        hb_thread.start()

        latest, last_rx, last_tick, last_stop_check = None, 0.0, 0.0, 0.0
        try:
            while True:
                try:
                    latest = sub.recv()
                    last_rx = time.time()
                except zmq.Again:
                    pass
                now = time.time()

                if self.session is not None and now - last_rx > self.args.idle_timeout:
                    print("[AI SERVER] C++ đã ngừng gửi frame — kết thúc phiên.")
                    self.end_session()
                    latest = None

                if latest is not None and now - last_tick >= self.period:
                    msg, latest = latest, None
                    last_tick = now
                    try:
                        out = self.process(msg)
                    except Exception as e:  # 1 nhịp lỗi không được làm sập server
                        out = None
                        if now - self._err_ts > 5:
                            print(f"[AI SERVER][LỖI] {e!r}")
                            self._err_ts = now
                    self._proc_ms.append((time.time() - now) * 1000)
                    if now - self._proc_report_ts >= 60 and self._proc_ms:   # để đối chiếu hiệu năng máy chạy thử
                        ms = sorted(self._proc_ms)
                        print(f"[AI SERVER] Thời gian xử lý 1 nhịp (60s qua): trung vị {ms[len(ms) // 2]:.0f} ms, "
                              f"chậm nhất {ms[-1]:.0f} ms, {len(ms)} nhịp.", flush=True)
                        self._proc_ms, self._proc_report_ts = [], now
                    if out is not None:
                        with pub_lock:
                            pub.send(out)
                            last_send[0] = time.time()

                if self.args.stop_file and now - last_stop_check >= 1.0:
                    last_stop_check = now
                    if os.path.exists(self.args.stop_file):
                        print("[AI SERVER] Nhận yêu cầu dừng — đóng phiên và thoát.")
                        break
        except KeyboardInterrupt:
            print("\n[AI SERVER] Nhận Ctrl+C — đang dừng...")
        finally:
            stop_hb.set()
            hb_thread.join(timeout=1.0)
            self.end_session()
            self.service.stop()
            sub.close(0)
            pub.close(0)
            ctx.term()


def main():
    lower_priority()
    AIServer(parse_args()).run()


if __name__ == "__main__":
    main()
