"""
session.py — Quản lý 1 PHIÊN giám sát cho AI server: log CSV, báo cáo JSON, ảnh bằng chứng.

Logic sự cố/bằng chứng giữ đúng như giamsatthi.AnalysisWorker._handle/_incident, chỉ khác là
trả về danh sách cảnh báo mới (seat_idx -> đường dẫn ảnh) để server báo sang C++ Client.
"""
import os
import time

import cv2

from ai_core import (Config, CheatingLogger, SessionReporter, capture_evidence_image,
                     finalize_daily_evidence, get_daily_evidence_dir, safe_folder_name)
from exam_ai_service import classify_behavior
from giamsatthi import LEVEL_TEXT, _seat_box

# Khoảng cách tối thiểu giữa 2 lần chụp bằng chứng của cùng 1 chỗ ngồi (giây)
EVIDENCE_MIN_INTERVAL = 3.0


class ExamSession:
    def __init__(self, room, proctor, log_dir):
        self.room = room
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"cheating_log_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        self.logger = CheatingLogger(log_file, Config.EVIDENCE_DIR)
        self.reporter = SessionReporter(Config.JSON_REPORT_FILE, Config.JSON_REPORT_DIR, room, proctor)
        self.evidence_dir = get_daily_evidence_dir(Config.EVIDENCE_DIR, room)
        self._last_shot = {}
        self.closed = False
        print(f"[Session] Bắt đầu phiên {self.reporter.session_id} | Phòng: {room} | Log: {log_file}")

    @property
    def session_id(self):
        return self.reporter.session_id

    def handle(self, snap, frame, seat_map):
        """Ghi log + mở/đóng sự cố cho 1 nhịp phân tích.

        Trả về {seat_idx: (evidence_path, is_new_incident)} của các cảnh báo mới ở nhịp này."""
        alerts = {}
        for v in snap.get("seats", []):
            vd, raw = v["verdict"], v["raw"]
            if vd is None:
                continue
            self.reporter.update_frame(raw["yaw"], raw["pitch"], raw["ear_l"], raw["ear_r"], raw["gaze"],
                                       vd["score"], vd["label"])
            self.logger.log(raw["yaw"], raw["pitch"], raw["ear_l"], raw["ear_r"], raw["gaze"], vd["score"],
                            vd["label"], vd["violations"], fraud_pct=vd["risk_pct"], confidence_level=vd["level"],
                            sustained_secs=vd["sustained"], person_id=v["idx"] + 1,
                            emotion=v["emotion"] or "", seat_label=v["label"])
            alert = self._incident(v, frame, seat_map)
            if alert:
                alerts[v["idx"]] = alert
        return alerts

    def _incident(self, v, frame, seat_map):
        vd, raw, idx = v["verdict"], v["raw"], v["idx"]
        active = self.reporter.has_active_incident(idx)
        if vd["is_cheating"] and vd["sustained"] >= Config.ALERT_SECONDS:
            if not active:
                self.reporter.open_incident(idx, time.strftime("%Y-%m-%d %H:%M:%S"), vd["violations"],
                                            raw["yaw"], raw["pitch"], raw["ear_l"], seat_label=v["label"])
                # Ghi hành vi lúc MỞ sự cố (cùng cách phân loại gửi cho C++) để báo cáo khớp với giao diện
                self.reporter._active_incidents[idx]["behavior_type"] = classify_behavior(
                    vd["violations"], True, v["bbox"] is not None)
            self.reporter.update_incident(idx, vd["risk_pct"], vd["level"], LEVEL_TEXT[vd["level"]],
                                          raw["yaw"], raw["pitch"], raw["ear_l"], vd["violations"])
            due = time.time() - self._last_shot.get(idx, 0.0) > EVIDENCE_MIN_INTERVAL
            if not active or (vd["risk_pct"] >= Config.FRAUD_CAPTURE_THRESHOLD and due):
                self._last_shot[idx] = time.time()
                box = v["bbox"] if v["bbox"] is not None else _seat_box(seat_map.seats[idx])
                img = capture_evidence_image(frame, vd["risk_pct"], LEVEL_TEXT[vd["level"]], vd["level"],
                                             vd["violations"], person_id=idx + 1,
                                             bbox=tuple(int(c) for c in box))
                name = f"{time.strftime('%Y%m%d_%H%M%S')}_{safe_folder_name(v['label'])}_fraud{int(vd['risk_pct'])}.jpg"
                path = os.path.abspath(os.path.join(self.evidence_dir, name))
                if cv2.imwrite(path, img):
                    self.reporter.set_incident_screenshot(idx, path)
                    return path, not active
        elif active:
            self.reporter.close_incident(idx)
        return None

    def reset_tracking(self):
        """Gọi khi sơ đồ chỗ ngồi bị học lại: chỉ số chỗ ngồi cũ không còn ý nghĩa."""
        self.reporter.close_all_incidents()
        self._last_shot.clear()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.reporter.close_all_incidents()
        self.logger.close()
        self.reporter.save()
        finalize_daily_evidence(Config.EVIDENCE_DIR, self.reporter.date_str, self.reporter.incidents,
                                self.reporter.session_id, self.room, self.reporter.proctor, room=self.room)
        print(f"[Session] Đã kết thúc phiên {self.reporter.session_id}: "
              f"{len(self.reporter.incidents)} sự cố, log và bằng chứng đã được lưu.")
