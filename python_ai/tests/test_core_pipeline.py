"""
Bộ kiểm thử tự động (Unit Tests) cho các thành phần cốt lõi của AI Server:
  1. Khử trùng lặp dữ liệu huấn luyện (train_core.load_labeled & video_group_keys)
  2. Mô hình SeatScorer (ràng buộc vật lý w >= 0, ngưỡng theo độ nhạy low/medium/high)
  3. Động cơ hành vi BehaviorEngine (cập nhật động sensitivity & absent_alert_seconds)
  4. Giao thức Protobuf FramePacket (sensitivity, absent_alert_seconds, seatmap_path)
"""

import os
import sys
import unittest
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_AI_DIR = os.path.dirname(THIS_DIR)
PROJECT_ROOT = os.path.dirname(PYTHON_AI_DIR)

for p in (PYTHON_AI_DIR, os.path.join(PYTHON_AI_DIR, "training"), os.path.join(PROJECT_ROOT, "proto", "python_out")):
    if p not in sys.path:
        sys.path.insert(0, p)

import train_core
from behavior import BehaviorEngine
from seat_scorer import SeatScorer, AGG_NAMES
import exam_monitor_pb2


class TestCorePipeline(unittest.TestCase):
    def test_1_deduplicate_labeled_dataset(self):
        """Kiểm tra khử trùng lặp .npz theo MD5(X) để chống rò rỉ dữ liệu."""
        import glob
        labeled_dir = os.path.join(PYTHON_AI_DIR, "training", "data", "labeled")
        paths = sorted(glob.glob(os.path.join(labeled_dir, "*.npz")))
        *_, raw_used = train_core.load_labeled(paths, 13, 20, deduplicate=False)
        *_, dedup_used = train_core.load_labeled(paths, 13, 20, deduplicate=True)
        self.assertGreaterEqual(len(raw_used), len(dedup_used))
        self.assertEqual(len(dedup_used), 12, "77 file .npz gốc phải gom về đúng 12 video độc lập")

        hashes = [train_core.x_hash(np.load(p)["X"]) for p in dedup_used]
        self.assertEqual(len(hashes), len(set(hashes)), "Mọi phần tử sau deduplicate phải có X duy nhất")

    def test_2_seat_scorer_constraints_and_thresholds(self):
        """Kiểm tra mô hình SeatScorer thỏa ràng buộc w >= 0 và thứ tự ngưỡng độ nhạy."""
        models_dir = os.path.join(PYTHON_AI_DIR, "models")
        scorer = SeatScorer.load_default(models_dir)
        self.assertIsNotNone(scorer, "Phải nạp được python_ai/models/seat_scorer.json")
        self.assertEqual(len(scorer.w), len(AGG_NAMES))

        # Mọi trọng số phải >= 0 (đơn điệu vật lý: tín hiệu lệch tăng -> điểm nghi vấn không giảm)
        for idx, weight in enumerate(scorer.w):
            self.assertGreaterEqual(
                float(weight), -1e-9, f"Trọng số đặc trưng #{idx} ({AGG_NAMES[idx]}) bị âm: {weight}"
            )

        # Các đặc trưng phụ thuộc góc đặt camera phải bằng 0
        for confounded_idx in (7, 8, 9, 10, 16):
            self.assertAlmostEqual(float(scorer.w[confounded_idx]), 0.0, places=6)

        # Ngưỡng độ nhạy: Cao (nhạy nhất, ngưỡng thấp nhất) < Trung bình < Thấp (ít báo sai nhất, ngưỡng cao nhất)
        thr_high = scorer.threshold_for("high")
        thr_med = scorer.threshold_for("medium")
        thr_low = scorer.threshold_for("low")
        self.assertLess(thr_high, thr_med)
        self.assertLess(thr_med, thr_low)

        # Kiểm tra tính đơn điệu trên cửa sổ 20x13 giả lập
        w_calm = np.zeros((20, 13), dtype=np.float32)
        w_calm[:, 0] = 1.0  # face_ok = 1
        w_active = w_calm.copy()
        w_active[:, 1] = 0.8  # yaw lớn
        w_active[:, 7] = 0.6  # dx lớn
        p_calm, _ = scorer.score(w_calm)
        p_active, _ = scorer.score(w_active)
        self.assertGreater(p_active, p_calm)

    def test_3_behavior_engine_dynamic_config(self):
        """Kiểm tra cập nhật động sensitivity và absent_alert_seconds trên BehaviorEngine."""
        models_dir = os.path.join(PYTHON_AI_DIR, "models")
        scorer = SeatScorer.load_default(models_dir)
        engine = BehaviorEngine(scorer=scorer)
        engine.set_runtime_config(sensitivity="high", absent_alert_seconds=6)
        self.assertAlmostEqual(engine.scorer_threshold, scorer.threshold_for("high"))
        self.assertEqual(engine.absent_alert, 6.0)

        vec = np.zeros(13, dtype=np.float32)
        raw = {"face_ok": 0.0, "yaw": 0.0, "pitch": 0.0, "gaze": 0.5, "dx": 0.0, "dw": -1.0, "dh": -1.0, "motion": 0.0}

        # Kiểm tra phát hiện rời ghế sau đúng 6 giây (thay vì 300 giây mặc định)
        v0 = engine.update(0, vec, raw, present=False, t=10.0)
        self.assertFalse(v0["is_cheating"])
        v1 = engine.update(0, vec, raw, present=False, t=15.0)  # 5s < 6s
        self.assertFalse(v1["is_cheating"])
        v2 = engine.update(0, vec, raw, present=False, t=16.5)  # 6.5s >= 6s
        self.assertTrue(v2["is_cheating"])
        self.assertTrue(any("Roi cho" in s for s in v2["violations"]))

    def test_4_protobuf_frame_packet_runtime_fields(self):
        """Kiểm tra đóng gói và giải mã các trường cấu hình động trong FramePacket."""
        pkt = exam_monitor_pb2.FramePacket()
        pkt.timestamp = 123456789
        pkt.frame_id = 42
        pkt.image_data = b"fake_jpeg"
        pkt.width = 1280
        pkt.height = 720
        pkt.exam_room = "PHONG_A1"
        pkt.proctor = "GiamThi01"
        pkt.reset_seq = 1
        pkt.sensitivity = "high"
        pkt.absent_alert_seconds = 20
        pkt.seatmap_path = "data/seatmap.json"

        raw = pkt.SerializeToString()
        parsed = exam_monitor_pb2.FramePacket()
        parsed.ParseFromString(raw)

        self.assertEqual(parsed.sensitivity, "high")
        self.assertEqual(parsed.absent_alert_seconds, 20)
        self.assertEqual(parsed.seatmap_path, "data/seatmap.json")
        self.assertEqual(parsed.exam_room, "PHONG_A1")


if __name__ == "__main__":
    unittest.main()
