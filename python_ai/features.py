"""
features.py — Vector đặc trưng theo TỪNG CHỖ NGỒI, dùng chung cho suy luận trực tiếp và trích dữ liệu huấn luyện.

13 chiều (đều không cần thấy mặt, mặt chỉ bổ sung khi có):
  0 face_ok | 1 yaw | 2 pitch | 3 roll | 4 ear_l | 5 ear_r | 6 gaze      (mặt, đã trừ baseline cá nhân)
  7 dx | 8 dy | 9 dw | 10 dh                                            (thân người so với vị trí ngồi chuẩn)
  11 speed | 12 motion                                                 (tốc độ dịch chuyển, mức cử động trong box)

Chỗ trống (không dò được người) được mã hoá bằng dw = dh = -1.
Lấy mẫu theo THỜI GIAN (FEATURE_HZ mẫu/giây) ở cả lúc huấn luyện lẫn lúc chạy thật để hai bên khớp nhau.
"""
import math

import cv2
import numpy as np

FEATURE_NAMES = ["face_ok", "yaw", "pitch", "roll", "ear_l", "ear_r", "gaze",
                 "dx", "dy", "dw", "dh", "speed", "motion"]
FEATURE_DIM = len(FEATURE_NAMES)

NO_FACE = {"face_ok": 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0,
           "ear_l": 0.30, "ear_r": 0.30, "gaze": 0.5}


def _clip(x, lo, hi):
    return float(min(max(x, lo), hi))


def compute_raw(seat, box, face, prev_anchor, motion, head_ratio=0.30):
    """Đại lượng thô (đơn vị vật lý) của 1 chỗ tại 1 thời điểm.
    box=None nghĩa là chỗ đang trống. face: dict hoặc None."""
    raw = dict(NO_FACE)
    if face is not None:
        raw.update(face_ok=1.0,
                   yaw=face["yaw"] - seat.get("yaw_base", 0.0),
                   pitch=face["pitch"] - seat.get("pitch_base", 0.0),
                   roll=face.get("roll", 0.0),
                   ear_l=face["ear_l"], ear_r=face["ear_r"], gaze=face["gaze"])
    bw, bh = max(seat["w"], 1.0), max(seat["h"], 1.0)
    if box is None:
        raw.update(dx=0.0, dy=0.0, dw=-1.0, dh=-1.0, speed=0.0, motion=0.0, present=0.0)
        return raw
    x0, y0, x1, y1 = box
    ax, ay = (x0 + x1) / 2.0, y0 + head_ratio * (y1 - y0)
    speed = 0.0 if prev_anchor is None else math.hypot(ax - prev_anchor[0], ay - prev_anchor[1]) / bw
    raw.update(dx=(ax - seat["ax"]) / bw, dy=(ay - seat["ay"]) / bh,
               dw=(x1 - x0) / bw - 1.0, dh=(y1 - y0) / bh - 1.0,
               speed=speed, motion=motion, present=1.0)
    return raw


def to_vector(raw):
    """Chuẩn hoá về khoảng nhỏ, ổn định cho Transformer."""
    v = np.zeros(FEATURE_DIM, dtype=np.float32)
    v[0] = raw["face_ok"]
    v[1] = _clip(raw["yaw"] / 90.0, -1, 1)
    v[2] = _clip(raw["pitch"] / 45.0, -1, 1)
    v[3] = _clip(raw["roll"] / 45.0, -1, 1)
    v[4] = _clip(raw["ear_l"] / 0.4, 0, 1)
    v[5] = _clip(raw["ear_r"] / 0.4, 0, 1)
    v[6] = _clip(raw["gaze"], 0, 1)
    v[7] = _clip(raw["dx"] / 2.0, -1, 1)
    v[8] = _clip(raw["dy"] / 2.0, -1, 1)
    v[9] = _clip(raw["dw"], -1, 1)
    v[10] = _clip(raw["dh"], -1, 1)
    v[11] = _clip(raw["speed"], 0, 1)
    v[12] = _clip(raw["motion"], 0, 1)
    return v


class FaceHold:
    """Giữ kết quả mặt cuối cùng trong thời gian ngắn để chống nhấp nháy khi mặt nhỏ dò lúc được lúc không.
    Giữ quá lâu sẽ che mất tín hiệu 'quay đầu đi nên mất mặt', nên mặc định chỉ ~1 nhịp lấy mẫu."""

    def __init__(self, hold_seconds=0.6):
        self.hold = hold_seconds
        self._last = {}          # seat_idx -> (t, face)

    def update(self, seat_idx, t, face):
        """Trả về (face_dùng_để_tính, mới_dò_được_lần_này)."""
        if face is not None:
            self._last[seat_idx] = (t, face)
            return face, True
        prev = self._last.get(seat_idx)
        if prev is not None and t - prev[0] <= self.hold:
            return prev[1], False
        self._last.pop(seat_idx, None)
        return None, False

    def clear(self):
        self._last.clear()


class MotionMeter:
    """Mức cử động trong từng box bằng chênh lệch ảnh xám giữa 2 lần lấy mẫu liên tiếp.
    Thay cho nhánh 3DCNN khi chưa có checkpoint: rẻ, chạy trên CPU, không cần thấy mặt."""

    def __init__(self, width=480, gain=6.0):
        self.width, self.gain = width, gain
        self._prev = None
        self._diff = None
        self._scale = 1.0

    def push(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        self._scale = self.width / float(w)
        small = cv2.resize(frame_bgr, (self.width, max(1, int(h * self._scale))))
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        self._diff = cv2.absdiff(gray, self._prev) if self._prev is not None and self._prev.shape == gray.shape else None
        self._prev = gray

    def energy(self, box):
        if self._diff is None or box is None:
            return 0.0
        s = self._scale
        x0, y0, x1, y1 = [int(round(c * s)) for c in box]
        H, W = self._diff.shape[:2]
        x0, x1 = max(0, x0), min(W, x1)
        y0, y1 = max(0, y0), min(H, y1)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        return _clip(float(self._diff[y0:y1, x0:x1].mean()) / 255.0 * self.gain, 0.0, 1.0)
