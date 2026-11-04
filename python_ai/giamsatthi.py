"""
giamsatthi.py — Giám sát trực tiếp cho camera đặt trên cao quan sát cả lớp (pipeline theo CHỖ NGỒI).

Luồng: (1) chính: đọc camera + vẽ; (2) AnalysisWorker: dò người -> chỗ ngồi -> mặt -> hành vi -> log/bằng chứng;
(3) EmotionWorker (DeepFace): CHỈ nhận ảnh khi thấy mặt. Vị trí (Hang-Cho) luôn có, kể cả khi không thấy mặt.
Giữ tương thích với run_thi.py / app_giamsat.py: main(args) với args.camera/room/proctor/fraud_threshold.
"""
import argparse
import json
import os
import threading
import time
from collections import deque

import cv2

import ai_core
from ai_core import (Config, CheatingLogger, SessionReporter, capture_evidence_image, draw_multi_hud,
                     finalize_daily_evidence, get_daily_evidence_dir, safe_folder_name)
from behavior import BehaviorEngine
from emotion_worker import EmotionWorker
from tsn_worker import TSNWorker
from face_analyzer import FaceAnalyzer
from person_detector import PersonDetector
from pipeline import SeatPipeline

LEVEL_COLOR = {"RAT CAO": (0, 0, 255), "CAO": (0, 60, 255), "TRUNG BINH": (0, 140, 255), "THAP": (0, 200, 255)}
LEVEL_TEXT = {"RAT CAO": "Gan chac chan gian lan - can thiep ngay", "CAO": "Rat nghi ngo - quan sat them",
              "TRUNG BINH": "Co dau hieu dang ngo - theo doi tiep", "THAP": "Dau hieu nhe - tiep tuc theo doi"}
NORMAL_COLOR, ABSENT_COLOR, OTHER_COLOR = (0, 200, 80), (140, 140, 140), (255, 170, 0)


def build_predictor():
    """Trả về (predict_fn hoặc None, ngưỡng). Không có model hợp lệ -> chạy bằng luật hành vi."""
    if not ai_core.TORCH_AVAILABLE:
        return None, Config.CHEAT_THRESHOLD
    model = ai_core.load_transformer_model()
    if model is None:
        return None, Config.CHEAT_THRESHOLD
    thr = Config.CHEAT_THRESHOLD
    try:
        with open(Config.MODEL_META_PATH, encoding="utf-8") as f:
            meta = json.load(f)
        ok = (meta.get("feature_dim") == Config.FEATURE_DIM and meta.get("seq_len") == Config.SEQ_LEN
              and abs(float(meta.get("feature_hz", 0)) - Config.FEATURE_HZ) < 1e-6)
        if not ok:
            print(f"[WARN] Model được huấn luyện với cấu hình khác ({meta}); bỏ qua Transformer, dùng luật hành vi.")
            return None, thr
        thr = float(meta.get("threshold", thr))
    except FileNotFoundError:
        print("[WARN] Không có transformer_seat_meta.json: dùng ngưỡng mặc định, không kiểm tra được cấu hình huấn luyện.")
    print(f"[OK] Transformer chỗ ngồi sẵn sàng (ngưỡng {thr:.2f}).")
    return (lambda seq: ai_core.classify_with_transformer(model, seq)[1]), thr


class AnalysisWorker(threading.Thread):
    """Luồng nền: nhận khung hình mới nhất, chạy 1 nhịp pipeline, ghi log/sự kiện/bằng chứng."""

    def __init__(self, pipeline, logger, reporter, evidence_dir):
        super().__init__(daemon=True, name="AnalysisWorker")
        self.pipeline, self.logger, self.reporter, self.evidence_dir = pipeline, logger, reporter, evidence_dir
        self.snapshot = None
        self.tick_times = deque(maxlen=20)
        self._frame, self._lock = None, threading.Lock()
        self._new, self._stop_ev = threading.Event(), threading.Event()
        self._t0 = time.time()
        self._last_shot = {}
        self._err_ts = 0.0
        self._paused = threading.Event()   # tạm dừng khi người dùng đang CHỈNH TAY chỗ ngồi — tránh luồng
                                            # nền vừa tự học/phân tích vừa bị sửa seat_map.seats cùng lúc

    def pause(self):
        self._paused.set()

    def resume(self):
        self._paused.clear()

    def submit(self, frame):
        with self._lock:
            self._frame = frame
        self._new.set()

    def stop(self):
        self._stop_ev.set()
        self._new.set()

    def hz(self):
        if len(self.tick_times) < 2:
            return 0.0
        return (len(self.tick_times) - 1) / max(self.tick_times[-1] - self.tick_times[0], 1e-6)

    def run(self):
        period = 1.0 / max(Config.FEATURE_HZ, 0.1)
        last = 0.0
        while not self._stop_ev.is_set():
            if self._paused.is_set():       # đang chỉnh tay chỗ ngồi ở luồng chính -> không đụng vào seat_map
                time.sleep(0.1)
                continue
            if not self._new.wait(0.2):
                continue
            wait = period - (time.time() - last)
            if wait > 0:
                time.sleep(wait)
            with self._lock:
                frame, self._frame = self._frame, None
                self._new.clear()
            if frame is None or self._stop_ev.is_set() or self._paused.is_set():
                continue
            last = time.time()
            try:
                snap = self.pipeline.tick(frame, last - self._t0)
                self.snapshot = snap
                self.tick_times.append(time.time())
                if snap.get("relearn_needed"):
                    print("[INFO] Phát hiện người mới ngồi ổn định ở vị trí chưa học -> tự động học lại "
                         "TOÀN BỘ chỗ ngồi (xoá cả tiến trình theo dõi của những người khác).")
                    self.pipeline.reset_all()
                    continue   # nhịp này không log/chấm điểm — chỗ ngồi vừa bị xoá, chưa có gì để tính
                self._handle(snap, frame)
            except Exception as e:                                   # không để luồng nền chết vì 1 nhịp lỗi
                if time.time() - self._err_ts > 5:
                    print(f"[AnalysisWorker][LỖI] {e!r}")
                    self._err_ts = time.time()

    def _handle(self, snap, frame):
        if snap.get("just_locked"):
            print(f"[INFO] Đã học xong {len(self.pipeline.seat_map.seats)} chỗ ngồi.")
        for v in snap["seats"]:
            vd, raw = v["verdict"], v["raw"]
            if vd is None:
                continue
            self.reporter.update_frame(raw["yaw"], raw["pitch"], raw["ear_l"], raw["ear_r"], raw["gaze"],
                                       vd["score"], vd["label"])
            self.logger.log(raw["yaw"], raw["pitch"], raw["ear_l"], raw["ear_r"], raw["gaze"], vd["score"],
                            vd["label"], vd["violations"], fraud_pct=vd["risk_pct"], confidence_level=vd["level"],
                            sustained_secs=vd["sustained"], person_id=v["idx"] + 1,
                            emotion=v["emotion"] or "", seat_label=v["label"])
            self._incident(v, frame)

    def _incident(self, v, frame):
        vd, raw, idx = v["verdict"], v["raw"], v["idx"]
        active = self.reporter.has_active_incident(idx)
        if vd["is_cheating"] and vd["sustained"] >= Config.ALERT_SECONDS:
            if not active:
                self.reporter.open_incident(idx, time.strftime("%Y-%m-%d %H:%M:%S"), vd["violations"],
                                            raw["yaw"], raw["pitch"], raw["ear_l"], seat_label=v["label"])
            self.reporter.update_incident(idx, vd["risk_pct"], vd["level"], LEVEL_TEXT[vd["level"]],
                                          raw["yaw"], raw["pitch"], raw["ear_l"], vd["violations"])
            due = time.time() - self._last_shot.get(idx, 0.0) > 3.0
            first = not active
            if first or (vd["risk_pct"] >= Config.FRAUD_CAPTURE_THRESHOLD and due):
                self._last_shot[idx] = time.time()
                box = v["bbox"] if v["bbox"] is not None else _seat_box(self.pipeline.seat_map.seats[idx])  # chỗ vắng: khoanh chỗ đó
                img = capture_evidence_image(frame, vd["risk_pct"], LEVEL_TEXT[vd["level"]], vd["level"],
                                             vd["violations"], person_id=idx + 1,
                                             bbox=tuple(int(c) for c in box))
                name = f"{time.strftime('%Y%m%d_%H%M%S')}_{safe_folder_name(v['label'])}_fraud{int(vd['risk_pct'])}.jpg"
                path = os.path.join(self.evidence_dir, name)
                cv2.imwrite(path, img)
                self.reporter.set_incident_screenshot(idx, path)
        elif active:
            self.reporter.close_incident(idx)


def _confirm_seatmap_keymap(key):
    """Quyết định THUẦN theo phím bấm ở màn hình xác nhận — tách riêng khỏi vòng lặp cv2 để kiểm thử
    được (cv2.waitKey/imshow cần màn hình thật, không chạy được trong môi trường kiểm thử tự động)."""
    if key in (13, 10):
        return "confirm"
    if key in (ord("r"), ord("R")):
        return "discard"
    if key in (ord("q"), ord("Q")):
        return "quit"
    return None


def _confirm_seatmap(cap, seat_map, win):
    """Hiện bản đồ chỗ ngồi ĐÃ NẠP đè lên camera TRỰC TIẾP, bắt xác nhận trước khi bắt đầu giám sát thật.
    Tránh âm thầm dùng bản đồ đã lệch (camera bị xê dịch, đổi phòng) mà không ai biết cho tới khi thấy
    sai lúc đang chạy — cùng vấn đề đã gặp và sửa ở công cụ smart_classroom_seatmap.py (dự án riêng).
    Trả về seat_map (đã xác nhận), None (bỏ, để pipeline tự học lại), hoặc "QUIT" (người dùng thoát hẳn)."""
    print(f"[INFO] Xác nhận bản đồ {len(seat_map.seats)} chỗ: ENTER = đúng, bắt đầu | R = sai, tự học lại | Q = thoát")
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        H, W = frame.shape[:2]
        k = min(1.0, 1400.0 / W)
        disp = cv2.resize(frame, (int(W * k), int(H * k))) if k < 1.0 else frame.copy()
        for s in seat_map.seats:
            x0, y0 = int((s["ax"] - s["w"] / 2) * k), int((s["ay"] - 0.3 * s["h"]) * k)
            x1, y1 = int((s["ax"] + s["w"] / 2) * k), int((s["ay"] + 0.7 * s["h"]) * k)
            cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 165, 255), 2)
            cv2.putText(disp, s["label"], (x0, max(15, y0 - 6)), Config.FONT, 0.5, (0, 165, 255), 1)
        cv2.putText(disp, f"XAC NHAN {len(seat_map.seats)} CHO DA LUU CO KHOP CAMERA HIEN TAI KHONG?",
                    (20, 30), Config.FONT, 0.6, (0, 165, 255), 2)
        cv2.putText(disp, "ENTER = Dung, bat dau giam sat  |  R = Sai, tu hoc lai  |  Q = Thoat",
                    (20, 58), Config.FONT, 0.55, (0, 165, 255), 2)
        cv2.imshow(win, disp)
        key = cv2.waitKey(1) & 0xFF
        action = _confirm_seatmap_keymap(key)
        if action == "confirm":
            print("[INFO] Đã xác nhận bản đồ chỗ ngồi — bắt đầu giám sát.")
            return seat_map
        if action == "discard":
            print("[INFO] Đã bỏ bản đồ đã nạp — hệ thống sẽ tự học lại chỗ ngồi từ đầu.")
            return None
        if action == "quit":
            return "QUIT"


def _finalize_manual_seats(pipeline, local_seats, out_path="seatmap.json", frame_size=None):
    """Khoá bản đồ chỗ ngồi vừa chỉnh tay vào pipeline đang chạy — tách riêng khỏi phần chuột/cv2 để
    kiểm thử được (cv2.imshow/waitKey cần màn hình thật, không chạy được trong môi trường kiểm thử).
    Trả về số chỗ đã khoá."""
    import seat_map as E
    E.relabel(local_seats, row_by_x=Config.SEAT_ROW_BY_X, col_right_to_left=Config.SEAT_COL_RIGHT_TO_LEFT,
              seat_far_first=getattr(Config, "SEAT_FAR_FIRST", False))
    pipeline.seat_map.seats = local_seats
    pipeline.seat_map.calibrating = False
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            data = pipeline.seat_map.to_dict()
            if frame_size:
                data["frame_size"] = [int(frame_size[0]), int(frame_size[1])]   # để nơi khác đối chiếu độ phân giải
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"[WARN] Không ghi được {out_path} ({e!r}) — vẫn dùng bản đồ vừa chỉnh cho phiên này.")
    return len(local_seats)


def _manual_seat_edit(cap, pipeline, worker, win):
    """Chỉnh tay chỗ ngồi NGAY TRONG phiên đang chạy — không cần video/tool riêng. Tạm dừng luồng phân
    tích nền trong lúc chỉnh (tránh vừa tự học vừa bị sửa seat_map.seats cùng lúc), cho chỉnh bằng chuột
    trên camera trực tiếp y hệt seat_map_editor.py, rồi khoá lại và chạy giám sát luôn khi lưu.
    S = lưu & chạy giám sát | C = xoá hết làm lại | Q/Esc = huỷ, quay về tự học như cũ."""
    import seat_map as E

    worker.pause()
    local_seats = [dict(s) for s in pipeline.seat_map.seats]   # bắt đầu từ chỗ đã học được (nếu có), không mất công
    state = {"sel": None, "drag": False, "k": 1.0}
    relabel_kwargs = dict(row_by_x=Config.SEAT_ROW_BY_X, col_right_to_left=Config.SEAT_COL_RIGHT_TO_LEFT,
                          seat_far_first=getattr(Config, "SEAT_FAR_FIRST", False))

    def on_mouse(event, x, y, flags, param):
        rx, ry = x / state["k"], y / state["k"]   # quy đổi toạ độ hiển thị (đã thu nhỏ) về toạ độ khung hình gốc
        if event == cv2.EVENT_LBUTTONDOWN:
            i = E.nearest_seat(local_seats, rx, ry, max_dist=0.6)
            if i is None:
                E.add_seat(local_seats, rx, ry)
                E.relabel(local_seats, **relabel_kwargs)
                state["sel"] = len(local_seats) - 1
            else:
                state["sel"], state["drag"] = i, True
        elif event == cv2.EVENT_MOUSEMOVE and state["drag"] and state["sel"] is not None:
            E.move_seat(local_seats, state["sel"], rx, ry)
        elif event == cv2.EVENT_LBUTTONUP:
            if state["drag"]:
                E.relabel(local_seats, **relabel_kwargs)
            state["drag"] = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            i = E.nearest_seat(local_seats, rx, ry, max_dist=0.6)
            if i is not None:
                E.remove_seat(local_seats, i)
                E.relabel(local_seats, **relabel_kwargs)
                state["sel"] = None

    cv2.setMouseCallback(win, on_mouse)
    print("[INFO] CHỈNH TAY chỗ ngồi: Trái=thêm/chọn/kéo | Phải=xoá | C=xoá hết | S=lưu & giám sát | Q=huỷ, quay lại tự học")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            H, W = frame.shape[:2]
            k = min(1.0, 1400.0 / W)
            state["k"] = k
            disp = cv2.resize(frame, (int(W * k), int(H * k))) if k < 1.0 else frame.copy()
            for i, s in enumerate(local_seats):
                x0, y0 = int((s["ax"] - s["w"] / 2) * k), int((s["ay"] - 0.3 * s["h"]) * k)
                x1, y1 = int((s["ax"] + s["w"] / 2) * k), int((s["ay"] + 0.7 * s["h"]) * k)
                color = (0, 255, 255) if i == state["sel"] else (0, 165, 255)
                cv2.rectangle(disp, (x0, y0), (x1, y1), color, 3 if i == state["sel"] else 2)
                cv2.putText(disp, s.get("label", "?"), (x0, max(14, y0 - 6)), Config.FONT, 0.5, color, 2)
            cv2.putText(disp, f"CHINH TAY CHO NGOI - {len(local_seats)} cho - Trai=them/keo Phai=xoa",
                        (20, 30), Config.FONT, 0.6, (0, 255, 255), 2)
            cv2.putText(disp, "C = xoa het | S = Luu & giam sat | Q = Huy, quay lai tu hoc",
                        (20, 58), Config.FONT, 0.55, (0, 255, 255), 2)
            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("s"), ord("S")):
                n = _finalize_manual_seats(pipeline, local_seats, frame_size=(W, H))
                print(f"[INFO] Đã lưu {n} chỗ -> seatmap.json, bắt đầu giám sát.")
                return
            if key in (ord("c"), ord("C")):
                local_seats.clear()
                state["sel"] = None
            if key in (ord("q"), ord("Q"), 27):
                print("[INFO] Đã huỷ chỉnh tay — quay lại tự học chỗ ngồi.")
                return
    finally:
        cv2.setMouseCallback(win, lambda *a: None)   # gỡ callback chỉnh tay, tránh ảnh hưởng vòng lặp giám sát chính
        worker.resume()


def _seat_box(seat):
    return (seat["ax"] - seat["w"] / 2, seat["ay"] - 0.3 * seat["h"], seat["ax"] + seat["w"] / 2, seat["ay"] + 0.7 * seat["h"])


def render(disp, snap, seat_map, k, fps, analysis_hz):
    """Vẽ kết quả lên khung hình hiển thị (tọa độ gốc nhân hệ số k)."""
    sc = lambda b: tuple(int(c * k) for c in b)
    states = {}
    if snap is None:
        cv2.putText(disp, "Dang khoi dong...", (30, 110), Config.FONT, 0.8, (0, 200, 255), 2)
    elif snap["calibrating"]:
        for o in snap["others"]:
            x0, y0, x1, y1 = sc(o["bbox"])
            cv2.rectangle(disp, (x0, y0), (x1, y1), OTHER_COLOR, 2)
        cv2.putText(disp, f"DANG HOC VI TRI CHO NGOI... {int(snap['progress'] * 100)}% (giu nguyen cho ngoi)",
                    (30, 110), Config.FONT, 0.8, (0, 200, 255), 2)
    else:
        for v in snap["seats"]:
            vd = v["verdict"]
            if v["present"]:
                state = "NORMAL"
                color = NORMAL_COLOR
                if vd and vd["is_cheating"]:
                    state = f"CHEATING [{vd['level']}]"
                    color = LEVEL_COLOR[vd["level"]]
                tail = (f" | {v['emotion']}" if v["emotion"] else " | dang phan tich cam xuc") if v.get("emo_ok", v["face_ok"]) \
                    else " | ..."
                states[v["idx"]] = {"bbox": sc(v["bbox"]), "label": f"[{v['label']}] {state}{tail}",
                                    "color": color, "is_cheating": bool(vd and vd["is_cheating"])}
                if v["face_bbox"] is not None:
                    fx0, fy0, fx1, fy1 = sc(v["face_bbox"])
                    cv2.rectangle(disp, (fx0, fy0), (fx1, fy1), (255, 255, 0), 1)
                elif v.get("emo_bbox") is not None:     # mặt do Haar dự phòng dò được (chỉ dùng cho cảm xúc)
                    fx0, fy0, fx1, fy1 = sc(v["emo_bbox"])
                    cv2.rectangle(disp, (fx0, fy0), (fx1, fy1), (255, 0, 255), 1)
            elif vd and vd["is_cheating"]:                         # chỗ trống quá lâu -> nghi ngờ
                states[v["idx"]] = {"bbox": sc(_seat_box(seat_map.seats[v["idx"]])),
                                    "label": f"[{v['label']}] VANG - {vd['violations'][0] if vd['violations'] else ''}",
                                    "color": (0, 0, 255), "is_cheating": True}
        for o in snap["others"]:
            x0, y0, x1, y1 = sc(o["bbox"])
            cv2.rectangle(disp, (x0, y0), (x1, y1), OTHER_COLOR, 2)
            cv2.putText(disp, o["label"], (x0, max(15, y0 - 6)), Config.FONT, 0.5, OTHER_COLOR, 1)
    draw_multi_hud(disp, states, fps)
    if snap is not None and not snap["calibrating"]:
        cv2.putText(disp, f"Thay mat: {snap['n_face']}/{snap['n_people']} | phan tich: {analysis_hz:.1f} Hz",
                    (32, 88), Config.FONT, 0.45, (180, 180, 180), 1)
    return disp


def main(args=None):
    args = args or parse_args()
    Config.CAMERA_ID = args.camera
    Config.EXAM_ROOM_ID = args.room
    Config.PROCTOR_NAME = args.proctor
    if args.fraud_threshold is not None:
        Config.FRAUD_CAPTURE_THRESHOLD = args.fraud_threshold
    stop_signal_path = getattr(Config, "STOP_SIGNAL_FILE", "stop.signal")
    if os.path.exists(stop_signal_path):
        try:
            os.remove(stop_signal_path)
        except OSError as e:
            print(f"[WARN] Không xóa được file tín hiệu dừng cũ: {e!r}")

    # Khởi động song song: nạp YOLO, dựng Face Mesh, mở camera và nạp Transformer (gồm `import torch`,
    # khá chậm) là 4 việc ĐỘC LẬP với nhau — trước đây làm TUẦN TỰ nên thời gian chờ cộng dồn cả 4. Giờ
    # chạy đồng thời bằng luồng riêng rồi gộp lại, tổng thời gian chờ ≈ việc chậm NHẤT thay vì cộng tất cả.
    t0 = time.time()
    print("[INFO] Đang khởi động song song: YOLO, Face Mesh, Transformer, camera...")
    _init = {}

    def _load_detector():
        _init["detector"] = PersonDetector(Config)

    def _load_faces():
        _init["faces"] = FaceAnalyzer(Config)

    def _load_predictor():
        _init["predict"], _init["thr"] = build_predictor()

    def _open_camera():
        cam_id = Config.CAMERA_ID
        if isinstance(cam_id, str) and cam_id.isdigit():
            cam_id = int(cam_id)
        c = cv2.VideoCapture(cam_id)
        if not c.isOpened() and isinstance(cam_id, int):
            c = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        c.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
        _init["cap"] = c

    def _load_tsn():
        w = TSNWorker(Config) if getattr(Config, 'TSN_ENABLED', True) else None
        if w:
            w.start()
            # Đợi TSN nạp model và hiển thị tick log sẵn sàng
            for _ in range(20):
                if w.ready:
                    break
                time.sleep(0.5)
        _init['tsn'] = w

    def _load_emotion():
        e = EmotionWorker(Config) if getattr(Config, 'EMOTION_ENABLED', True) else None
        if e:
            e.start()
        _init['emotion'] = e

    threads = [threading.Thread(target=f, daemon=True)
               for f in (_load_detector, _load_faces, _load_predictor, _open_camera, _load_tsn, _load_emotion)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"[INFO] Khởi động xong sau {time.time() - t0:.1f}s.")

    detector, faces, predict, thr, cap, tsn, emotion = (
        _init["detector"], _init["faces"], _init["predict"],
        _init["thr"], _init["cap"], _init.get("tsn"), _init.get("emotion")
    )
    if not detector.ready:
        print("[LỖI] Không có detector người nên không thể giám sát. Cài: pip install ultralytics")
        return
    if not cap.isOpened():
        print(f"[LỖI] Không mở được camera: {Config.CAMERA_ID}")
        return

# TSN va Emotion da duoc nap song song trong _init
    engine = BehaviorEngine(Config, predict, Config.SEQ_LEN, thr)
    seat_map = None
    seatmap_path = getattr(args, "seatmap", None)   # an toàn khi main() được gọi từ run_thi.py (Namespace
                                                      # riêng, không phải parse_args() của chính file này)
    if seatmap_path:
        import json
        from seat_map import SeatMap
        with open(seatmap_path, encoding="utf-8") as f:
            seat_map, note = SeatMap.from_dict_scaled(
                json.load(f), cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), Config)
        print(f"[INFO] Đã nạp bản đồ {len(seat_map.seats)} chỗ từ {seatmap_path} — cần xác nhận trước khi bắt đầu.")
        if note:
            print(f"[WARN] {note}")

    win = "GIAM SAT THI"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # Có bản đồ nạp sẵn (--seatmap) -> bắt xác nhận còn khớp camera HIỆN TẠI không, TRƯỚC khi chạy thật.
    # Không có (tự học từ đầu như trước) -> bỏ qua bước này, giữ nguyên hành vi cũ.
    if seat_map is not None:
        seat_map = _confirm_seatmap(cap, seat_map, win)
        if seat_map == "QUIT":
            cap.release()
            cv2.destroyAllWindows()
            if emotion:
                emotion.stop()
            if tsn:
                tsn.stop()
            return

    pipeline = SeatPipeline(Config, detector, faces, emotion, engine, seat_map, tsn=tsn)

    logger = CheatingLogger(Config.LOG_FILE, Config.EVIDENCE_DIR)
    reporter = SessionReporter(Config.JSON_REPORT_FILE, Config.JSON_REPORT_DIR, Config.EXAM_ROOM_ID, Config.PROCTOR_NAME)
    evidence_dir = get_daily_evidence_dir(Config.EVIDENCE_DIR, Config.EXAM_ROOM_ID)
    worker = AnalysisWorker(pipeline, logger, reporter, evidence_dir)
    worker.start()

    print(f"[INFO] Bắt đầu giám sát. Học chỗ ngồi {Config.SEAT_CALIBRATION_SECONDS:.0f}s đầu (mọi người ngồi yên). "
          f"Phím: Q dừng | R học lại chỗ ngồi | E chỉnh tay chỗ ngồi.")
    fps, t_prev, fails = 0.0, time.time(), 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                fails += 1
                if fails > 50:
                    print("[LỖI] Camera không trả khung hình nữa.")
                    break
                time.sleep(0.05)
                continue
            fails = 0
            worker.submit(frame)
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now
            H, W = frame.shape[:2]
            k = min(1.0, 1400.0 / W)
            disp = cv2.resize(frame, (int(W * k), int(H * k))) if k < 1.0 else frame.copy()
            render(disp, worker.snapshot, pipeline.seat_map, k, fps, worker.hz())
            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")) or os.path.exists(stop_signal_path):
                break
            if key in (ord("r"), ord("R")):
                pipeline.reset_all()
                print("[INFO] Học lại chỗ ngồi.")
            if key in (ord("e"), ord("E")):
                _manual_seat_edit(cap, pipeline, worker, win)
    finally:
        worker.stop()
        worker.join(timeout=3.0)
        if emotion:
            emotion.stop()
        if tsn:
            tsn.stop()
        cap.release()
        cv2.destroyAllWindows()
        logger.close()
        reporter.save()
        finalize_daily_evidence(Config.EVIDENCE_DIR, reporter.date_str, reporter.incidents, reporter.session_id,
                                Config.EXAM_ROOM_ID, reporter.proctor, room=Config.EXAM_ROOM_ID)
        if os.path.exists(stop_signal_path):
            try:
                os.remove(stop_signal_path)
            except OSError:
                pass
        print("[INFO] Đã dừng giám sát. Toàn bộ log và bằng chứng đã được lưu trữ.")


def parse_args():
    p = argparse.ArgumentParser(description="Giám sát thi cử — camera trên cao")
    p.add_argument("--camera", default=getattr(Config, "CAMERA_ID", 0))
    p.add_argument("--room", type=str, default=getattr(Config, "EXAM_ROOM_ID", "PHONG_01"))
    p.add_argument("--proctor", type=str, default=getattr(Config, "PROCTOR_NAME", "GiamThi"))
    p.add_argument("--fraud-threshold", type=float, default=None)
    p.add_argument("--seatmap", default=None, help="File JSON từ seat_map_editor.py — dùng thẳng, bỏ qua tự học chỗ ngồi")
    p.add_argument("--frame-width", type=int, default=None)
    p.add_argument("--frame-height", type=int, default=None)
    a = p.parse_args()
    if a.frame_width:
        Config.FRAME_WIDTH = a.frame_width
    if a.frame_height:
        Config.FRAME_HEIGHT = a.frame_height
    return a


if __name__ == "__main__":
    main()