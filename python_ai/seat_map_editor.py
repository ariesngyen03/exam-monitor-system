r"""
seat_map_editor.py — Xem và CHỈNH TAY bản đồ chỗ ngồi trên 1 khung hình thật, trước khi chạy giám sát.

Giải quyết đúng vấn đề: hệ thống tự học chỉ nhận ra chỗ nào CÓ NGƯỜI lúc học — nếu học sinh vắng/đến trễ
đúng lúc đó, chỗ đó có thể bị bỏ sót dù đã có "suy luận theo lưới" (seat_map.py) hỗ trợ phần lớn trường hợp.
Công cụ này cho bạn XÁC NHẬN BẰNG MẮT toàn bộ chỗ ngồi thật của phòng, chỉ cần làm 1 lần cho mỗi cách sắp phòng.

    python seat_map_editor.py --video C:\...\29.09.01.mp4                     # học tự động rồi cho chỉnh tay
    python seat_map_editor.py --video C:\...\29.09.01.mp4 --load seatmap.json # mở lại bản đã lưu để sửa tiếp

Thao tác:
  Click TRÁI vào chỗ TRỐNG   -> thêm 1 chỗ mới tại đó
  Click TRÁI vào chỗ ĐÃ CÓ   -> chọn (viền vàng), kéo để DI CHUYỂN
  Click PHẢI vào chỗ ĐÃ CÓ   -> XOÁ chỗ đó
  Phím S  -> LƯU ra file JSON (--out, mặc định seatmap.json)
  Phím R  -> học lại tự động từ đầu (bỏ hết chỉnh tay)
  Phím Q  -> thoát (không lưu nếu chưa bấm S)

Dùng file đã lưu: truyền --seatmap <file> cho check_pipeline_video.py, extract_features_video.py, giamsatthi.py
để BỎ QUA bước tự học, dùng thẳng bản đồ đã xác nhận bằng mắt.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

import seat_map as E
from seat_map import SeatMap

WIN = "Chinh ban do cho ngoi"
COLOR_REAL, COLOR_INFER, COLOR_MANUAL, COLOR_SEL = (0, 200, 0), (0, 165, 255), (255, 255, 0), (0, 255, 255)


class Editor:
    def __init__(self, frame, seats, cfg=None):
        self.frame, self.seats = frame, seats
        self.sel, self.drag = None, False
        self.dirty = False
        # Đọc từ cfg để chỉnh tay cũng gán nhãn Dãy/Ghế đúng kiểu phòng (VD dãy kéo dài theo chiều sâu),
        # khớp với đúng cấu hình seat_map.py dùng khi tự học — không đổi gì nếu cfg=None (mặc định cũ).
        g = (lambda k, d: getattr(cfg, k, d)) if cfg is not None else (lambda k, d: d)
        self._relabel_kwargs = dict(row_by_x=g("SEAT_ROW_BY_X", False),
                                    col_right_to_left=g("SEAT_COL_RIGHT_TO_LEFT", False),
                                    seat_far_first=g("SEAT_FAR_FIRST", False))

    def _relabel(self):
        E.relabel(self.seats, **self._relabel_kwargs)

    def draw(self):
        img = self.frame.copy()
        for i, s in enumerate(self.seats):
            x0, y0 = int(s["ax"] - s["w"] / 2), int(s["ay"] - 0.3 * s["h"])
            x1, y1 = int(s["ax"] + s["w"] / 2), int(s["ay"] + 0.7 * s["h"])
            color = COLOR_SEL if i == self.sel else (COLOR_MANUAL if s.get("manual") else (COLOR_INFER if s.get("inferred") else COLOR_REAL))
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 3 if i == self.sel else 2)
            cv2.putText(img, s.get("label", "?"), (x0, max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        h, w = img.shape[:2]
        legend = ["Xanh la: hoc duoc that su | Cam: suy luan tu luoi | Vang nhat: ban them tay",
                 "Trai vao cho trong = them | Trai vao cho co = chon/keo | Phai = xoa | S = luu | R = hoc lai | Q = thoat"]
        for i, t in enumerate(legend):
            cv2.putText(img, t, (10, h - 34 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, f"So cho: {len(self.seats)}" + (" *chua luu*" if self.dirty else ""),
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return img

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            i = E.nearest_seat(self.seats, x, y, max_dist=0.6)
            if i is None:
                idx = E.add_seat(self.seats, x, y)
                self._relabel()
                self.sel, self.dirty = idx, True
            else:
                self.sel, self.drag = i, True
        elif event == cv2.EVENT_MOUSEMOVE and self.drag and self.sel is not None:
            E.move_seat(self.seats, self.sel, x, y)
            self.dirty = True
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drag:
                self._relabel()
            self.drag = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            i = E.nearest_seat(self.seats, x, y, max_dist=0.6)
            if i is not None:
                E.remove_seat(self.seats, i)
                self._relabel()
                self.sel, self.dirty = None, True


def _open_capture(source):
    if isinstance(source, str) and source.strip().isdigit():
        idx = int(source.strip())
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
        return cap, True
    return cv2.VideoCapture(source), False


def learn_initial(video, cfg):
    from face_analyzer import FaceAnalyzer
    from person_detector import PersonDetector
    det, faces = PersonDetector(cfg), FaceAnalyzer(cfg)
    sm = SeatMap(cfg)
    cap, is_cam = _open_capture(video)
    t, last_frame = 0.0, None
    while sm.calibrating and t <= sm.calib_seconds * 2 + 5:
        if not is_cam:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break
        last_frame = frame
        dets = det.detect(frame)
        extras = [{"yaw": f["yaw"], "pitch": f["pitch"]} if (f := faces.analyze(frame, d["bbox"])) else None for d in dets]
        sm.update_calibration(t, dets, extras)
        if is_cam:
            time.sleep(0.2)
        t += 0.5
    if sm.calibrating and last_frame is not None:
        sm.finalize(t)
    cap.release()
    return last_frame, sm.seats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="0", help="Đường dẫn file video hoặc chỉ số camera (0, 1...)")
    ap.add_argument("--load", default=None, help="Mở lại 1 file seatmap.json đã lưu để sửa tiếp thay vì học lại")
    ap.add_argument("--out", default="seatmap.json")
    a = ap.parse_args()
    from ai_core import Config

    if a.load and os.path.exists(a.load):
        with open(a.load, encoding="utf-8") as f:
            data = json.load(f)
        seats = data["seats"]
        cap, _ = _open_capture(a.video)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            w, h = data.get("frame_size", [1280, 720])
            frame = np.full((int(h), int(w), 3), 30, dtype=np.uint8)
    else:
        print("[INFO] Đang học tự động, vài chục giây...")
        frame, seats = learn_initial(a.video, Config)
        if frame is None:
            raise RuntimeError(f"Không đọc được nguồn video/camera: {a.video}")
        print(f"[INFO] Học tự động được {len(seats)} chỗ. Giờ bạn xem và chỉnh tay nếu thiếu/sai.")

    ed = Editor(frame, seats, Config)
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, ed.on_mouse)
    while True:
        cv2.imshow(WIN, ed.draw())
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("q"), ord("Q")):
            if ed.dirty:
                print("[LƯU Ý] Còn thay đổi CHƯA LƯU. Bấm S trước khi Q nếu muốn giữ lại.")
            break
        if k in (ord("s"), ord("S")):
            with open(a.out, "w", encoding="utf-8") as f:
                json.dump({"seats": ed.seats, "frame_size": [int(ed.frame.shape[1]), int(ed.frame.shape[0])]},
                          f, ensure_ascii=False, indent=2)
            ed.dirty = False
            print(f"[ĐÃ LƯU] {len(ed.seats)} chỗ -> {a.out}")
        if k in (ord("r"), ord("R")):
            print("[INFO] Học lại từ đầu...")
            frame2, seats2 = learn_initial(a.video, Config)
            if frame2 is not None:
                ed.frame, ed.seats, ed.sel, ed.dirty = frame2, seats2, None, True
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
