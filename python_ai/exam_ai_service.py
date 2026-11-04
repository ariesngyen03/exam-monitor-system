"""
exam_ai_service.py — Headless AI Inference Engine for ExamMonitorSystem.

Đóng gói toàn bộ logic suy luận (Person Detection, Seat Mapping, Face Mesh,
Transformer 13D, Behavior Rules, Emotion) thành một Service độc lập (Headless),
không chứa giao diện đồ họa, không mở camera OpenCV, sẵn sàng tích hợp qua IPC ZeroMQ/Protobuf.
"""
import os
import re
import json
import time
import numpy as np
import cv2

import ai_core
from ai_core import Config
from person_detector import PersonDetector
from face_analyzer import FaceAnalyzer
from behavior import BehaviorEngine
from seat_map import SeatMap
from pipeline import SeatPipeline
from emotion_worker import EmotionWorker
from seat_scorer import SeatScorer
import giamsatthi


def parse_seat_label_to_id(seat_label: str, fallback_idx: int) -> int:
    """Chuyển nhãn 'D1-G2' thành desk_id số nguyên (102) cho Protobuf int32."""
    m = re.match(r"D(\d+)-G(\d+)", seat_label or "")
    if m:
        try:
            return int(m.group(1)) * 100 + int(m.group(2))
        except ValueError:
            pass
    return fallback_idx + 1


# Thứ tự ưu tiên khi 1 chỗ có nhiều vi phạm cùng lúc: (từ khoá trong violations, behavior_type gửi C++).
# Từ khoá khớp đúng chuỗi do behavior.rule_evaluate()/BehaviorEngine.update() sinh ra.
_BEHAVIOR_PRIORITY = [
    ("sang cho khac", "Leaning"),
    ("quay dau", "Turning Head"),
    ("nhin lech", "Turning Head"),
    ("cui/ngua dau", "Head Down"),
    ("nghieng nguoi", "Leaning"),
    ("roi cho", "Absent"),
]


def classify_behavior(violations, is_cheating: bool, present: bool) -> str:
    """Rút gọn danh sách vi phạm (tiếng Việt không dấu) thành 1 behavior_type cho C++ Client."""
    if not is_cheating:
        return "Normal" if present else "Empty"
    viols = " ".join(violations or []).lower()
    for key, behavior in _BEHAVIOR_PRIORITY:
        if key in viols:
            return behavior
    return "Absent" if not present else "Suspicious"


class ExamAIService:
    """Dịch vụ AI suy luận không giao diện (Headless Service) cho hệ thống giám sát thi."""

    def __init__(self, cfg=None, seatmap_path=None, enable_emotion=True):
        self.cfg = cfg or Config
        self.enable_emotion = enable_emotion

        print("[ExamAIService] Đang khởi tạo các module AI...")
        t0 = time.time()

        # 1. Detector người (YOLOv8)
        self.detector = PersonDetector(self.cfg)
        if not self.detector.ready:
            print("[ExamAIService][CẢNH BÁO] PersonDetector chưa sẵn sàng!")

        # 2. Phân tích khuôn mặt (FaceAnalyzer)
        self.faces = FaceAnalyzer(self.cfg)

        # 3. Transformer phân loại gian lận (13 chiều)
        self.predict_fn, self.threshold = giamsatthi.build_predictor()

        # 4. Engine hành vi: SeatScorer quyết định (nếu có model), luật giải thích, Transformer ghi chú thêm
        self.scorer = None
        if getattr(self.cfg, "SEAT_SCORER_ENABLED", True):
            self.scorer = SeatScorer.load_default(os.path.join(ai_core.BASE_DIR, "models"))
        sens = getattr(self.cfg, "SEAT_SCORER_SENSITIVITY", "medium")
        scorer_thr = self.scorer.threshold_for(sens) if self.scorer else None
        if self.scorer:
            print(f"[ExamAIService] SeatScorer: độ nhạy '{sens}' (ngưỡng {scorer_thr:.3f}).")
        else:
            print("[ExamAIService] Không có models/seat_scorer.json — quyết định bằng luật hành vi + Transformer.")
        self.engine = BehaviorEngine(
            self.cfg, self.predict_fn, self.cfg.SEQ_LEN, self.threshold,
            scorer=self.scorer, scorer_threshold=scorer_thr,
        )

        # 5. Bản đồ chỗ ngồi (SeatMap)
        self.seat_map = None
        if seatmap_path and os.path.exists(seatmap_path):
            try:
                with open(seatmap_path, "r", encoding="utf-8") as f:
                    self.seat_map, note = SeatMap.from_dict_scaled(
                        json.load(f), self.cfg.FRAME_WIDTH, self.cfg.FRAME_HEIGHT, self.cfg
                    )
                print(f"[ExamAIService] Đã nạp bản đồ chỗ ngồi từ: {seatmap_path}")
            except Exception as e:
                print(f"[ExamAIService][LỖI] Không nạp được {seatmap_path}: {e}")

        if self.seat_map is None:
            self.seat_map = SeatMap(self.cfg)

        # 6. Nhận diện cảm xúc (DeepFace worker chạy ngầm)
        self.emotion = None
        if self.enable_emotion and getattr(self.cfg, "EMOTION_ENABLED", True):
            try:
                self.emotion = EmotionWorker(self.cfg)
                self.emotion.start()
                print("[ExamAIService] EmotionWorker đã được kích hoạt chạy ngầm.")
            except Exception as e:
                print(f"[ExamAIService][CẢNH BÁO] Không thể khởi động EmotionWorker: {e}")
                self.emotion = None

        # 7. Nhánh TSN (mặc định tắt để đảm bảo hiệu năng CPU thời gian thực)
        self.tsn = None

        # 8. Pipeline điều phối nhịp phân tích
        self.pipeline = SeatPipeline(
            self.cfg, self.detector, self.faces, self.emotion, self.engine, self.seat_map, self.tsn
        )

        self.current_sensitivity = sens
        self.current_absent_alert = int(getattr(self.cfg, "SEAT_ABSENT_ALERT_SECONDS", 300))
        self.current_seatmap_path = (
            os.path.abspath(seatmap_path) if seatmap_path and os.path.exists(seatmap_path) else ""
        )

        print(f"[ExamAIService] Khởi tạo hoàn tất sau {time.time() - t0:.2f}s.")

    def update_runtime_config(self, sensitivity: str = "", absent_alert_seconds: int = 0,
                              seatmap_path: str = "") -> bool:
        """Cập nhật cấu hình động từ gói tin FramePacket của C++ Client.
        Trả về True nếu bản đồ chỗ ngồi vừa được nạp lại / đổi chế độ."""
        sens = (sensitivity or "").strip().lower()
        if sens in ("low", "medium", "high") and sens != self.current_sensitivity:
            self.current_sensitivity = sens
            self.cfg.SEAT_SCORER_SENSITIVITY = sens
            self.engine.set_runtime_config(sensitivity=sens)
            if self.scorer is not None:
                print(f"[ExamAIService] Cập nhật độ nhạy AI -> '{sens}' (ngưỡng {self.engine.scorer_threshold:.3f}).",
                      flush=True)

        if absent_alert_seconds > 0 and int(absent_alert_seconds) != self.current_absent_alert:
            self.current_absent_alert = int(absent_alert_seconds)
            self.cfg.SEAT_ABSENT_ALERT_SECONDS = float(absent_alert_seconds)
            self.engine.set_runtime_config(absent_alert_seconds=float(absent_alert_seconds))
            print(f"[ExamAIService] Cập nhật thời gian cảnh báo rời chỗ -> {self.current_absent_alert}s.", flush=True)

        norm_path = os.path.abspath(seatmap_path.strip()) if (seatmap_path or "").strip() else ""
        if norm_path != self.current_seatmap_path:
            if norm_path and os.path.exists(norm_path):
                try:
                    with open(norm_path, "r", encoding="utf-8") as f:
                        new_map, _ = SeatMap.from_dict_scaled(
                            json.load(f), self.cfg.FRAME_WIDTH, self.cfg.FRAME_HEIGHT, self.cfg
                        )
                    self.seat_map = new_map
                    self.pipeline.seat_map = new_map
                    self.pipeline.engine._st.clear()
                    self.current_seatmap_path = norm_path
                    print(f"[ExamAIService] Đã nạp bản đồ chỗ ngồi cố định từ: {norm_path}", flush=True)
                    return True
                except Exception as e:
                    print(f"[ExamAIService][LỖI] Không nạp được {norm_path}: {e}", flush=True)
            elif not norm_path and self.current_seatmap_path:
                self.current_seatmap_path = ""
                self.seat_map = SeatMap(self.cfg)
                self.pipeline.seat_map = self.seat_map
                self.pipeline.reset_all()
                print("[ExamAIService] Đã chuyển về chế độ tự động học sơ đồ chỗ ngồi.", flush=True)
                return True
        return False

    def reset_seatmap(self):
        """Học lại toàn bộ sơ đồ chỗ ngồi từ đầu."""
        print("[ExamAIService] Yêu cầu học lại sơ đồ chỗ ngồi...")
        self.pipeline.reset_all()

    def stop(self):
        """Dừng tài nguyên và các luồng ngầm."""
        if self.emotion is not None:
            self.emotion.stop()
        if self.tsn is not None:
            self.tsn.stop()
        print("[ExamAIService] Đã dọn dẹp tài nguyên AI.")

    def process_frame(self, frame_bgr: np.ndarray, timestamp_sec: float) -> dict:
        """
        Xử lý 1 khung hình thời gian thực.
        
        Args:
            frame_bgr: Khung hình OpenCV BGR (từ C++ gửi sang qua IPC).
            timestamp_sec: Dấu thời gian tính theo giây (float).
            
        Returns:
            dict chứa:
                calibrating (bool): Đang trong giai đoạn học chỗ ngồi hay không
                progress (float): Tiến độ học (0.0 -> 1.0)
                detections (list): Danh sách chỗ ngồi/người (dict, khớp message Detection;
                                   thêm "seat_idx" để ghép với snapshot)
                n_people (int): Số lượng người phát hiện
                n_faces (int): Số lượng khuôn mặt phát hiện
                relearned (bool): Nhịp này vừa kích hoạt học lại sơ đồ chỗ ngồi (bỏ qua ghi log)
                just_locked (bool): Nhịp này vừa học xong sơ đồ chỗ ngồi
                snapshot (dict): Kết quả thô của SeatPipeline.tick()
        """
        snap = self.pipeline.tick(frame_bgr, timestamp_sec)

        relearned = bool(snap.get("relearn_needed"))
        if relearned:
            print("[ExamAIService] Phát hiện người mới ổn định, tự động học lại sơ đồ chỗ ngồi.")
            self.pipeline.reset_all()

        is_calibrating = snap.get("calibrating", False)
        calib_progress = snap.get("progress", 1.0)
        detections = []

        def _other(i, o, behavior):
            return {
                "track_id": 1000 + i,          # không trùng chỉ số chỗ ngồi
                "desk_id": 0,
                "seat_label": o.get("label", ""),
                "behavior_type": behavior,
                "confidence": 0.0,
                "bbox": [float(b) for b in o["bbox"]],
                "is_present": True,
                "is_cheating": False,
                "violations": [],
                "emotion": "",
                "face_bbox": [],
                "risk_level": "",
                "seat_idx": None,
            }

        if is_calibrating or snap.get("just_locked"):
            # Trong giai đoạn học (kể cả nhịp vừa học xong): báo các box người phát hiện được để C++ vẽ tạm
            for i, o in enumerate(snap.get("others", [])):
                if o.get("bbox"):
                    detections.append(_other(i, o, "Calibrating"))
        elif not relearned:
            # Giai đoạn giám sát chính thức theo chỗ ngồi
            seats = self.pipeline.seat_map.seats
            for v in snap.get("seats", []):
                idx = v["idx"]
                seat_label = v["label"]
                present = v["present"] and v["bbox"] is not None
                verdict = v.get("verdict") or {}
                is_cheating = bool(verdict.get("is_cheating", False))
                risk_pct = float(verdict.get("risk_pct", 0.0))
                violations = list(verdict.get("violations", []))

                if present:
                    box = v["bbox"]
                else:
                    # Chỗ vắng: khoanh vị trí ghế đã học
                    seat_info = seats[idx] if idx < len(seats) else None
                    box = giamsatthi._seat_box(seat_info) if seat_info else (0.0, 0.0, 0.0, 0.0)
                    ox, oy = self.pipeline.cam_offset          # bù xê dịch camera cho khung chỗ trống
                    box = (box[0] + ox, box[1] + oy, box[2] + ox, box[3] + oy)

                detections.append({
                    "track_id": idx + 1,
                    "desk_id": parse_seat_label_to_id(seat_label, idx),
                    "seat_label": seat_label,
                    "behavior_type": classify_behavior(violations, is_cheating, present),
                    # Điểm chú ý của SeatScorer (0..1) nếu có, dùng để XẾP HẠNG chỗ cần quan sát; không phải xác suất gian lận
                    "confidence": (verdict.get("attention") if verdict.get("attention") is not None
                                   else risk_pct / 100.0),
                    "bbox": [float(b) for b in box],
                    "is_present": present,
                    "is_cheating": is_cheating,
                    "violations": violations,
                    "emotion": (v.get("emotion") or "") if present else "",
                    "face_bbox": [float(b) for b in (v.get("face_now_bbox") or ())] if present else [],
                    "risk_level": verdict.get("level", "") if is_cheating else "",
                    "seat_idx": idx,
                })
            # Người chưa khớp chỗ nào đã học (VD giám thị đi lại)
            for i, o in enumerate(snap.get("others", [])):
                if o.get("bbox"):
                    detections.append(_other(i, o, "Unassigned"))

        return {
            "calibrating": is_calibrating,
            "progress": calib_progress,
            "detections": detections,
            "n_people": snap.get("n_people", 0),
            "n_faces": snap.get("n_face", 0),
            "relearned": relearned,
            "just_locked": bool(snap.get("just_locked")),
            "snapshot": snap,   # dữ liệu thô của pipeline (raw/verdict) cho ghi log & bằng chứng
        }
