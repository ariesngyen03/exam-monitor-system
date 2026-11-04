"""
pipeline.py — Một "nhịp" phân tích cho camera trên cao. Dùng chung cho chạy trực tiếp (giamsatthi.py)
và trích dữ liệu huấn luyện (extract_features_video.py) để đặc trưng hai bên khớp nhau.

Mỗi nhịp:  dò người -> (học hoặc ghép) chỗ ngồi -> tìm mặt trong vùng đầu -> đặc trưng 13 chiều
           -> [luật + Transformer] -> [cảm xúc CHỈ khi có mặt].
Vị trí (nhãn chỗ ngồi) luôn lấy từ box thân người nên có ở cả hai trường hợp thấy mặt / không thấy mặt.
"""
import math
import statistics

from features import FaceHold, MotionMeter, compute_raw, to_vector
from seat_map import SeatMap, anchor_of


class SeatPipeline:
    def __init__(self, cfg, detector, face_analyzer=None, emotion=None, engine=None, seat_map=None, tsn=None):
        self.tsn = tsn
        self.cfg, self.detector = cfg, detector
        self.faces, self.emotion, self.engine = face_analyzer, emotion, engine
        self.seat_map = seat_map if seat_map is not None else SeatMap(cfg)
        self.motion = MotionMeter()
        self.hold = FaceHold(getattr(cfg, "FACE_HOLD_SECONDS", 0.6) if cfg is not None else 0.6)
        self.prev_anchor = {}
        self._had_face = set()
        self._emo_rr = 0
        self._disp_box = {}   # box đã làm mượt (EMA) CHỈ để hiển thị — không dùng cho tính đặc trưng/luật
        self.bbox_smooth = getattr(cfg, "SEAT_BBOX_SMOOTH_EMA", 0.4) if cfg is not None else 0.4
        # Tự động học lại TOÀN BỘ chỗ ngồi khi có người mới (không khớp chỗ nào đã học) đứng/ngồi ỔN ĐỊNH
        # đủ lâu ở CÙNG 1 vị trí — không phải chỉ xuất hiện thoáng qua 1 lần phân tích (tránh học lại liên
        # tục mỗi khi ai đó đi ngang qua khung hình, VD giám thị kiểm tra phòng).
        g = (lambda k, d: getattr(cfg, k, d)) if cfg is not None else (lambda k, d: d)
        self.newcomer_enabled = g("SEAT_NEWCOMER_RELEARN_ENABLED", True)
        self.newcomer_relearn_seconds = g("SEAT_NEWCOMER_RELEARN_SECONDS", 8.0)
        self.newcomer_stable_radius = g("SEAT_NEWCOMER_STABLE_RADIUS_W", 0.6)
        self._newcomer_pos = None
        self._newcomer_since = None
        # Bù xê dịch camera: nếu MỌI người cùng lệch 1 hướng so với sơ đồ đã học thì đó là camera bị xô/rung,
        # không phải hành vi -> dịch cả sơ đồ theo, tránh báo "nghiêng người"/ghép nhầm chỗ hàng loạt.
        self.cam_comp_enabled = g("CAM_SHIFT_COMPENSATION", True)
        self.cam_min_people = g("CAM_SHIFT_MIN_PEOPLE", 3)
        self.cam_relearn_seconds = g("CAM_SHIFT_RELEARN_SECONDS", 20.0)
        self.cam_offset = (0.0, 0.0)
        self._cam_shift_since = None

    # ── 1 nhịp ─────────────────────────────────────────────────────────────
    def tick(self, frame, t):
        dets = self.detector.detect(frame)
        self.motion.push(frame)
        if self.seat_map.calibrating:
            return self._calibrating(frame, t, dets)
        return self._running(frame, t, dets)

    def _smoothed_box(self, idx, box):
        """Làm mượt vị trí box CHỈ để hiển thị (nhãn/mũi tên đỡ nhấp nháy khi người ngồi yên nhưng box
        dò được rung nhẹ giữa các lần phân tích) — KHÔNG dùng giá trị này để tính đặc trưng/luật hành vi,
        chỗ đó vẫn dùng box thô để không làm mất tín hiệu chuyển động thật (đứng dậy, cúi hẳn xuống...)."""
        if box is None:
            self._disp_box.pop(idx, None)
            return None
        prev = self._disp_box.get(idx)
        if prev is None:
            new_box = box
        else:
            a = self.bbox_smooth
            new_box = tuple(a * b + (1 - a) * p for b, p in zip(box, prev))
        self._disp_box[idx] = new_box
        return new_box

    def _track_newcomer(self, t, others):
        """others: người dò được nhưng KHÔNG khớp chỗ nào đã học. Trả về True đúng lúc cần học lại toàn bộ
        (đã đứng yên đủ SEAT_NEWCOMER_RELEARN_SECONDS ở cùng 1 vị trí)."""
        if not self.newcomer_enabled:
            return False
        if not others:
            self._newcomer_pos = None
            self._newcomer_since = None
            return False
        match = None
        if self._newcomer_pos is not None:
            for o in others:
                x0, y0, x1, y1 = o["bbox"]
                cx, cy = (x0 + x1) / 2.0, y0 + self.seat_map.head_ratio * (y1 - y0)
                w = max(x1 - x0, 1.0)
                if math.hypot(cx - self._newcomer_pos[0], cy - self._newcomer_pos[1]) <= self.newcomer_stable_radius * w:
                    match = (cx, cy)
                    break
        if match is not None:
            self._newcomer_pos = match             # vẫn là người đó -> cập nhật vị trí bám theo, GIỮ mốc thời gian
        else:
            x0, y0, x1, y1 = others[0]["bbox"]
            cx, cy = (x0 + x1) / 2.0, y0 + self.seat_map.head_ratio * (y1 - y0)
            self._newcomer_pos, self._newcomer_since = (cx, cy), t   # người/vị trí khác -> đếm lại từ đầu
        return (t - self._newcomer_since) >= self.newcomer_relearn_seconds

    def _update_cam_offset(self, dets, assign, t):
        """Ước lượng độ xê dịch chung của camera từ các cặp (người, chỗ) đã ghép. Trả về True nếu camera
        lệch lớn kéo dài -> nên học lại sơ đồ chỗ ngồi."""
        sm = self.seat_map
        if not self.cam_comp_enabled or not sm.seats:
            return False
        ox, oy = self.cam_offset
        res = []
        for d, idx in zip(dets, assign):
            if idx is not None:
                ax, ay = anchor_of(d["bbox"], sm.head_ratio)
                s = sm.seats[idx]
                res.append((ax - (s["ax"] + ox), ay - (s["ay"] + oy)))
        med_w = statistics.median(s["w"] for s in sm.seats)
        if len(res) >= max(self.cam_min_people, (len(sm.seats) + 1) // 2):
            mx = statistics.median(r[0] for r in res)
            my = statistics.median(r[1] for r in res)
            spread = statistics.median(math.hypot(r[0] - mx, r[1] - my) for r in res)
            # Lệch ĐỒNG LOẠT (độ phân tán nhỏ) và đủ lớn -> camera dịch; cập nhật dần để không giật
            if spread < 0.25 * med_w and math.hypot(mx, my) > 0.15 * med_w:
                self.cam_offset = (ox + 0.7 * mx, oy + 0.7 * my)
        shifted = math.hypot(*self.cam_offset) > 0.35 * med_w
        if not shifted:
            self._cam_shift_since = None
            return False
        if self._cam_shift_since is None:
            self._cam_shift_since = t
            print(f"[SeatPipeline] Camera xê dịch ~({self.cam_offset[0]:.0f}, {self.cam_offset[1]:.0f}) px — "
                  f"đang bù; sẽ học lại sơ đồ nếu kéo dài {self.cam_relearn_seconds:.0f}s.")
        return (t - self._cam_shift_since) >= self.cam_relearn_seconds

    def reset_all(self):
        """Học lại TOÀN BỘ chỗ ngồi và xoá sạch mọi trạng thái đang theo dõi (dùng chung cho phím 'R' và
        cho việc tự động kích hoạt khi có người mới ổn định). CẢNH BÁO: xoá cả tiến trình theo dõi của
        những người đang ngồi ổn định (thời gian nghi ngờ, thời gian vắng mặt...), không chỉ người mới."""
        self.seat_map.reset()
        if self.engine is not None:
            self.engine._st.clear()
        self.prev_anchor.clear()
        self._had_face.clear()
        self._disp_box.clear()
        self.hold.clear()
        self._newcomer_pos = None
        self._newcomer_since = None
        self.cam_offset = (0.0, 0.0)
        self._cam_shift_since = None

    def _face(self, frame, bbox):
        return self.faces.analyze(frame, bbox) if (self.faces is not None and bbox is not None) else None

    def _calibrating(self, frame, t, dets):
        faces = [self._face(frame, d["bbox"]) for d in dets]
        extras = [{"yaw": f["yaw"], "pitch": f["pitch"]} if f else None for f in faces]
        done = self.seat_map.update_calibration(t, dets, extras)
        return {"t": t, "calibrating": not done, "progress": self.seat_map.progress(t), "seats": [],
                "others": [{"bbox": d["bbox"], "label": self.seat_map.label(None)} for d in dets],
                "n_people": len(dets), "n_face": sum(f is not None for f in faces), "just_locked": done,
                "relearn_needed": False}

    def _running(self, frame, t, dets):
        sm = self.seat_map
        assign = sm.assign(dets, self.cam_offset)
        cam_relearn = self._update_cam_offset(dets, assign, t)
        ox, oy = self.cam_offset
        by_seat, others = {}, []
        for d, idx in zip(dets, assign):
            if idx is None:
                others.append({"bbox": d["bbox"], "label": sm.label(None)})
            else:
                by_seat[idx] = d
        views, fresh = [], []
        for seat in sm.seats:
            idx = seat["idx"]
            d = by_seat.get(idx)
            box = d["bbox"] if d else None
            face_raw = self._face(frame, box)
            if box is None:
                face, is_fresh = None, False
                self.hold.update(idx, t, None)
            else:
                face, is_fresh = self.hold.update(idx, t, face_raw)
            # Dự phòng cảm xúc: Face Mesh không thấy mặt nhưng Haar (bộ dò của ứng dụng cảm xúc gốc) thấy -> vẫn lấy
            # được ảnh mặt cho DeepFace. CHỈ dùng cho cảm xúc, KHÔNG đổi face_ok/đặc trưng/luật hành vi.
            emo_face = None
            if box is not None and face is None and self.emotion is not None and hasattr(self.faces, "haar_face"):
                emo_face = self.faces.haar_face(frame, box)
            seat_now = seat if (ox == 0.0 and oy == 0.0) else dict(seat, ax=seat["ax"] + ox, ay=seat["ay"] + oy)
            raw = compute_raw(seat_now, box, face, self.prev_anchor.get(idx), self.motion.energy(box), sm.head_ratio)
            if box is not None:
                self.prev_anchor[idx] = anchor_of(box, sm.head_ratio)
            vec = to_vector(raw)
            verdict = self.engine.update(idx, vec, raw, box is not None, t) if self.engine else None
            tsn_feat = None
            if self.tsn is not None:
                if box is not None:
                    x0, y0, x1, y1 = [int(v) for v in box]
                    h_img, w_img = frame.shape[:2]
                    x0, y0 = max(0, x0), max(0, y0)
                    x1, y1 = min(w_img, x1), min(h_img, y1)
                    if x1 > x0 and y1 > y0:
                        self.tsn.push_frame(idx, frame[y0:y1, x0:x1])
                    tsn_feat = self.tsn.get_features(idx)
                else:
                    self.tsn.clear(idx)
            view = {"idx": idx, "label": seat["label"], "present": box is not None, "bbox": self._smoothed_box(idx, box),
                    "face_ok": face is not None, "face_bbox": face["bbox"] if face else None,
                    "emo_ok": face is not None or emo_face is not None,
                    "emo_bbox": emo_face["bbox"] if emo_face else None,
                    # Mặt dò được Ở NHỊP NÀY (không tính mặt đang giữ tạm) -> C++ cắt đúng vùng mặt cho nhận diện cảm xúc FER+
                    "face_now_bbox": (face_raw or emo_face or {}).get("bbox"), "emotion": None, "tsn_feat": tsn_feat, "raw": raw, "vec": vec, "verdict": verdict}
            views.append(view)
            if is_fresh and face.get("img") is not None:
                fresh.append((idx, face["img"]))
            elif emo_face is not None:
                fresh.append((idx, emo_face["img"]))
        self._emotion_step(views, fresh)
        if self.engine is not None and self.emotion is not None:
            raw_label = getattr(self.emotion, "get_smoothed", None)
            for v in views:     # cờ cảm xúc lo lắng kéo dài (chỉ tham khảo) — cần cảm xúc của nhịp này nên gắn sau
                # Như bản gốc: cờ tính trên nhãn đã làm mượt (mode), không chờ bước giữ ổn định hiển thị 5 giây.
                lab = (raw_label(v["idx"]) if (raw_label and v["emo_ok"]) else v["emotion"])
                self.engine.apply_emotion(v["idx"], v["verdict"], lab, t)
        relearn_needed = self._track_newcomer(t, others) or cam_relearn
        return {"t": t, "calibrating": False, "progress": 1.0, "seats": views, "others": others,
                "n_people": len(dets), "n_face": sum(v["face_ok"] for v in views), "just_locked": False,
                "relearn_needed": relearn_needed}

    # ── cảm xúc: CHỈ khi có mặt ────────────────────────────────────────────
    def _emotion_step(self, views, fresh):
        if self.emotion is None:
            return
        if fresh:                                    # mỗi nhịp chỉ gửi 1 người (xoay vòng) để không quá tải
            fresh.sort(key=lambda x: x[0])
            pick = next((f for f in fresh if f[0] >= self._emo_rr), fresh[0])
            self.emotion.submit_face(pick[0], pick[1])
            self._emo_rr = pick[0] + 1
        for v in views:
            if v["emo_ok"]:
                self._had_face.add(v["idx"])
                v["emotion"] = self.emotion.get_emotion(v["idx"], None)
            elif v["idx"] in self._had_face:         # mất mặt: xoá cảm xúc cũ, không hiển thị/không suy đoán
                self.emotion.clear(v["idx"])
                self._had_face.discard(v["idx"])
