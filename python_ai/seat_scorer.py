"""
seat_scorer.py — Chấm điểm bất thường cho 1 chỗ ngồi từ cửa sổ đặc trưng 10s (SEQ_LEN x 13, xem features.py).

Dùng CHUNG cho huấn luyện (training/train_seat_scorer.py) và chạy thật (behavior.BehaviorEngine) để đặc trưng tổng
hợp ở hai bên chắc chắn giống nhau. Mô hình: Logistic Regression -> trọng số đọc được, giải thích được.
"""
import json
import os

import numpy as np

# Chỉ số trong vector 13 chiều của features.to_vector()
_FACE, _YAW, _PITCH, _EAR_L, _GAZE, _DX, _DY, _DW, _DH, _SPEED, _MOTION = 0, 1, 2, 4, 6, 7, 8, 9, 10, 11, 12

AGG_NAMES = [
    "yaw_abs_last", "yaw_abs_mean4", "yaw_abs_max",
    "pitch_abs_mean4",
    "dx_abs_last", "dx_abs_mean4", "dx_std",
    "dy_last", "dy_mean4",
    "dw_abs_last", "dh_abs_last",
    "gaze_dev_mean4", "gaze_std", "ear_std",
    "motion_mean4", "speed_mean4",
    "face_frac",
]

# Lời giải thích (tiếng Việt không dấu như các vi phạm khác) cho tín hiệu đóng góp lớn nhất
_EXPLAIN = {
    "yaw_abs": "Quay dau nhieu", "pitch_abs": "Cui/ngua dau", "dx": "Nghieng nguoi sang ben",
    "dy": "Thay doi tu the (len/xuong)", "dw": "Thay doi tu the lon", "dh": "Thay doi tu the lon",
    "gaze": "Nhin lech sang ben", "ear": "Mat dao dong bat thuong", "motion": "Cu dong nhieu",
    "speed": "Di chuyen nhanh", "face": "Khuon mat bi che/khuat",
}


def aggregate(X):
    """X: (N, L, 13) hoặc (L, 13) -> đặc trưng tổng hợp (N, len(AGG_NAMES))."""
    X = np.asarray(X, np.float32)
    if X.ndim == 2:
        X = X[None]
    present = ~((X[:, :, _DW] <= -0.999) & (X[:, :, _DH] <= -0.999))
    last4 = slice(-4, None)
    face = X[:, :, _FACE] > 0.5

    def masked_mean(v, m):
        return np.where(m.sum(1) > 0, (v * m).sum(1) / np.maximum(m.sum(1), 1), 0.0)

    yaw = np.abs(X[:, :, _YAW])
    f = [
        yaw[:, -1], masked_mean(yaw[:, last4], face[:, last4]), (yaw * face).max(1),
        masked_mean(np.abs(X[:, last4, _PITCH]), face[:, last4]),
        np.abs(X[:, -1, _DX]), np.abs(X[:, last4, _DX]).mean(1),
        np.where(present.all(1), X[:, :, _DX].std(1), 0.0),
        X[:, -1, _DY], X[:, last4, _DY].mean(1),
        np.abs(X[:, -1, _DW]) * present[:, -1], np.abs(X[:, -1, _DH]) * present[:, -1],
        masked_mean(np.abs(X[:, last4, _GAZE] - 0.5), face[:, last4]),
        X[:, :, _GAZE].std(1), X[:, :, _EAR_L].std(1),
        X[:, last4, _MOTION].mean(1), X[:, last4, _SPEED].mean(1),
        face.mean(1),
    ]
    return np.stack(f, 1).astype(np.float32)


class SeatScorer:
    """Nạp models/seat_scorer.json; score(window) -> (xác suất bất thường, lời giải thích tín hiệu chính)."""

    def __init__(self, path):
        with open(path, encoding="utf-8") as fh:
            m = json.load(fh)
        if m.get("features") != AGG_NAMES:
            raise ValueError("seat_scorer.json được huấn luyện với bộ đặc trưng khác — chạy lại train_seat_scorer.py")
        self.mean = np.array(m["mean"], np.float32)
        self.std = np.array(m["std"], np.float32)
        self.w = np.array(m["w"], np.float32)
        self.b = float(m["b"])
        self.threshold = float(m["threshold"])
        self.thresholds = {k: float(v) for k, v in m.get("thresholds", {}).items()}
        self.seq_len = int(m.get("seq_len", 20))

    @classmethod
    def load_default(cls, models_dir):
        path = os.path.join(models_dir, "seat_scorer.json")
        if not os.path.exists(path):
            return None
        try:
            return cls(path)
        except Exception as e:
            print(f"[SeatScorer][WARN] Không nạp được {path}: {e}")
            return None

    def threshold_for(self, sensitivity):
        """'low' | 'medium' | 'high' -> ngưỡng (low = ít báo sai nhất, high = bắt nhiều nhất)."""
        return self.thresholds.get(sensitivity, self.threshold)

    def score(self, window):
        z = (aggregate(window)[0] - self.mean) / self.std
        contrib = z * self.w
        p = 1.0 / (1.0 + np.exp(-np.clip(contrib.sum() + self.b, -30, 30)))
        top = AGG_NAMES[int(np.argmax(contrib))]
        reason = next((txt for key, txt in _EXPLAIN.items() if top.startswith(key)), "Tu the bat thuong")
        return float(p), reason
