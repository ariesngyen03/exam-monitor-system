"""
behavior.py — Luật hành vi + chống nhiễu + hợp nhất Transformer cho từng chỗ ngồi.

Không import ai_core; Transformer được truyền vào dưới dạng hàm predict_fn(chuỗi[SEQ_LEN, D]) -> xác suất CHEATING.
Ngưỡng luật là giá trị khởi đầu BAN ĐẦU, chưa được kiểm chứng trên dữ liệu thật của bạn: chỉnh trong Config sau khi xem kết quả.
"""
from collections import deque

import numpy as np


def _g(cfg, k, d):
    return getattr(cfg, k, d) if cfg is not None else d


def rule_evaluate(raw, cfg=None):
    """Điểm nghi ngờ (0..~12) và danh sách vi phạm từ đại lượng thô của 1 chỗ đang có người.
    Luật về mặt chỉ áp dụng khi thấy mặt; luật về thân người áp dụng mọi lúc."""
    score, viol = 0, []
    yaw_t = _g(cfg, "SEAT_YAW_THRESHOLD", 30.0)
    pitch_t = _g(cfg, "SEAT_PITCH_THRESHOLD", 25.0)
    if raw.get("face_ok", 0.0) >= 0.5:
        if abs(raw["yaw"]) > yaw_t:
            score += 5 if abs(raw["yaw"]) > 2 * yaw_t else 3
            viol.append(f"Quay dau {'TRAI' if raw['yaw'] < 0 else 'PHAI'} ({raw['yaw']:.0f})")
        if abs(raw["pitch"]) > pitch_t:
            score += 3
            viol.append(f"Cui/ngua dau bat thuong ({raw['pitch']:.0f})")
        if raw["gaze"] < 0.2 or raw["gaze"] > 0.8:
            score += 2
            viol.append("Nhin lech sang ben")
    lean = abs(raw["dx"])
    if lean > _g(cfg, "SEAT_LEAN_STRONG", 1.2):
        score += 5
        viol.append(f"Nghieng/sang cho khac ({raw['dx']:+.1f})")
    elif lean > _g(cfg, "SEAT_LEAN_MILD", 0.7):
        score += 3
        viol.append(f"Nghieng nguoi sang ben ({raw['dx']:+.1f})")
    size_t = _g(cfg, "SEAT_SIZE_CHANGE", 0.35)
    if abs(raw["dw"]) > size_t or abs(raw["dh"]) > size_t:
        score += 2
        viol.append("Thay doi tu the lon (dung day/cui sat)")
    if raw["motion"] > _g(cfg, "SEAT_MOTION_HIGH", 0.6):
        score += 2
        viol.append("Cu dong nhieu")
    return score, viol


def debounce(prev, cheat_streak, normal_streak, raw_label, enter_n, exit_n):
    if raw_label == "CHEATING":
        cheat_streak, normal_streak = cheat_streak + 1, 0
    else:
        cheat_streak, normal_streak = 0, normal_streak + 1
    if prev != "CHEATING" and cheat_streak >= enter_n:
        prev = "CHEATING"
    elif prev == "CHEATING" and normal_streak >= exit_n:
        prev = "NORMAL"
    return prev, cheat_streak, normal_streak


def risk_from(score, n_viol, sustained):
    base = min(score / 10.0, 1.0) * 65.0
    bonus = min(sustained / 8.0, 1.0) * 20.0
    combo = min(n_viol * 3.0, 15.0)
    pct = min(base + bonus + combo, 100.0)
    level = "RAT CAO" if pct >= 85 else "CAO" if pct >= 65 else "TRUNG BINH" if pct >= 40 else "THAP"
    return round(pct, 1), level


class _SeatState:
    def __init__(self, seq_len):
        self.buf = deque(maxlen=seq_len)
        self.prob = 0.0
        self.has_prob = False
        self.cheat_streak = self.normal_streak = 0
        self.label = "NORMAL"
        self.cheat_start = None
        self.absent_since = None
        self.stress_since = None   # từ lúc nào cảm xúc đang hiển thị thuộc nhóm "lo lắng/căng thẳng"
        self.attention = None      # điểm chú ý của SeatScorer (0..1, đã làm mượt)


class BehaviorEngine:
    def __init__(self, cfg=None, predict_fn=None, seq_len=None, threshold=None, scorer=None, scorer_threshold=None):
        self.cfg = cfg
        self.predict_fn = predict_fn
        # SeatScorer (seat_scorer.py): khi có, QUYẾT ĐỊNH cần chú ý dựa trên điểm của nó (đánh giá leave-one-session-out
        # tốt hơn luật viết tay — xem training/train_seat_scorer.py); luật vẫn chạy để sinh LỜI GIẢI THÍCH.
        self.scorer = scorer
        self.scorer_threshold = scorer_threshold if scorer_threshold is not None else (
            scorer.threshold if scorer is not None else None)
        self.seq_len = seq_len or _g(cfg, "SEQ_LEN", 20)
        self.threshold = threshold if threshold is not None else _g(cfg, "CHEAT_THRESHOLD", 0.6)
        self.ema = _g(cfg, "SEAT_PROB_EMA", 0.5)
        # Chống nhiễu BẤT ĐỐI XỨNG theo yêu cầu: báo CHEATING gần như NGAY (enter mặc định 1 nhịp — không
        # chờ), nhưng phải NORMAL liên tục đủ SEAT_EXIT_SECONDS (mặc định 5 giây) mới chuyển hiển thị về
        # NORMAL — tránh chớp tắt trạng thái khi hành vi dao động quanh ngưỡng. Tính theo GIÂY, tự quy đổi
        # ra số nhịp dựa theo FEATURE_HZ nên không phụ thuộc tốc độ lấy mẫu.
        hz = _g(cfg, "FEATURE_HZ", 2.0)
        self.enter_n = max(1, round(_g(cfg, "SEAT_ENTER_SECONDS", 0.0) * hz))
        self.exit_n = max(1, round(_g(cfg, "SEAT_EXIT_SECONDS", 5.0) * hz))
        self.absent_alert = _g(cfg, "SEAT_ABSENT_ALERT_SECONDS", 300.0)
        self.attn_ema = _g(cfg, "SEAT_SCORER_EMA", 0.6)
        self.fusion = _g(cfg, "HYBRID_FUSION_MODE", "or")
        # Cờ cảm xúc lo lắng/căng thẳng kéo dài (thuật toán cảm xúc của hệ thống gốc): CHỈ để tham khảo, không
        # cộng điểm, không đổi nhãn CHEATING/NORMAL, không ảnh hưởng risk_pct.
        self.stress_labels = set(_g(cfg, "EMOTION_STRESS_LABELS", ["fear", "sad", "angry"]))
        self.stress_sustain = _g(cfg, "EMOTION_STRESS_SUSTAIN_SECONDS", 3.0)
        self._st = {}

    def state(self, idx):
        if idx not in self._st:
            self._st[idx] = _SeatState(self.seq_len)
        return self._st[idx]

    def set_runtime_config(self, sensitivity=None, absent_alert_seconds=None):
        """Cập nhật động độ nhạy của SeatScorer và thời gian cảnh báo vắng mặt ngay trong lúc chạy."""
        if sensitivity in ("low", "medium", "high") and self.scorer is not None:
            self.scorer_threshold = self.scorer.threshold_for(sensitivity)
        if absent_alert_seconds is not None and absent_alert_seconds > 0:
            self.absent_alert = float(absent_alert_seconds)

    def update(self, idx, vec, raw, present, t):
        st = self.state(idx)

        if present:
            if st.absent_since is not None:
                # Vừa rời chỗ rồi quay lại -> RESET toàn bộ trạng thái theo dõi, nhận diện lại từ đầu.
                # Buffer/chuỗi cũ (có đoạn lúc vắng mặt) không còn ý nghĩa, tiếp tục dùng sẽ cho dự đoán sai.
                # QUAN TRỌNG: clear() rồi mới append vec của nhịp này — clear() TRƯỚC khi thêm, không phải
                # sau, nếu không sẽ xoá mất luôn vector vừa thêm.
                st.buf.clear()
                st.has_prob = False
                st.prob = 0.0
                st.cheat_streak = st.normal_streak = 0
                st.label = "NORMAL"
                st.cheat_start = None
                st.stress_since = None
                st.attention = None
            st.absent_since = None
            score, viol = rule_evaluate(raw, self.cfg)
        else:
            if st.absent_since is None:
                st.absent_since = t
            gone = t - st.absent_since
            score, viol = (5, [f"Roi cho {gone:.0f}s"]) if gone >= self.absent_alert else (0, [])
        st.buf.append(vec)   # LUÔN thêm SAU khối reset ở trên (nếu có) — không được thêm trước khi clear()
        rb_raw = "CHEATING" if score >= 5 else "NORMAL"

        tf_raw, prob = "NORMAL", None
        if self.predict_fn is not None and present and len(st.buf) == self.seq_len:
            p = float(self.predict_fn(np.stack(st.buf)))
            st.prob = p if not st.has_prob else self.ema * p + (1 - self.ema) * st.prob
            st.has_prob = True
        if self.predict_fn is not None and st.has_prob:
            prob = st.prob
            tf_raw = "CHEATING" if prob >= self.threshold else "NORMAL"

        reason = None
        if self.scorer is not None and present and len(st.buf) == self.seq_len:
            p_s, reason = self.scorer.score(np.stack(st.buf))
            st.attention = p_s if st.attention is None else self.attn_ema * p_s + (1 - self.attn_ema) * st.attention

        # Tín hiệu THÔ (chưa chống nhiễu) từ việc hợp nhất rule-based + Transformer theo đúng nhịp này.
        if self.scorer is not None and present:
            # SeatScorer quyết định; Transformer (nếu có) chỉ thêm ghi chú, luật chỉ để giải thích
            raw_combined = ("CHEATING" if st.attention is not None and st.attention >= self.scorer_threshold
                            else "NORMAL")
        elif self.predict_fn is None or not st.has_prob:
            raw_combined = rb_raw
        elif self.fusion == "and":
            raw_combined = "CHEATING" if (rb_raw == "CHEATING" and tf_raw == "CHEATING") else "NORMAL"
        else:
            raw_combined = "CHEATING" if (rb_raw == "CHEATING" or tf_raw == "CHEATING") else "NORMAL"

        # Chống nhiễu áp dụng vào QUYẾT ĐỊNH CUỐI CÙNG (đã hợp nhất) — không chỉ riêng rule-based như
        # trước — nên dù CHEATING đến từ rule-based hay Transformer đều được báo ngay, và NORMAL đều cần
        # giữ ổn định đủ lâu mới hiển thị lại.
        final_label, st.cheat_streak, st.normal_streak = debounce(
            st.label, st.cheat_streak, st.normal_streak, raw_combined, self.enter_n, self.exit_n)
        cheating = final_label == "CHEATING"

        # Transformer chỉ đóng góp ghi chú/điểm khi KHÔNG có SeatScorer (chế độ dự phòng) — tránh để mô hình
        # đối chứng (F1 ~0.45) đẩy mức độ cảnh báo lên khi SeatScorer đã là bộ quyết định.
        if self.scorer is None and tf_raw == "CHEATING" and cheating:
            viol = list(viol) + ["Transformer: hanh vi bat thuong"]
            score = max(score, int(round(prob * 10)))
        if self.scorer is not None and cheating and present and st.attention is not None:
            if not viol and reason:
                viol = [f"Mo hinh: {reason}"]   # luật không thấy gì cụ thể -> nêu tín hiệu chính của mô hình
            score = max(score, int(round(st.attention * 10)))
        st.label = final_label
        if cheating:
            if st.cheat_start is None:
                st.cheat_start = t
        else:
            st.cheat_start = None
        sustained = 0.0 if st.cheat_start is None else t - st.cheat_start
        pct, level = risk_from(score, len(viol), sustained)
        return {"label": st.label, "is_cheating": cheating, "score": score, "violations": viol,
                "prob": prob, "rule_label": rb_raw, "risk_pct": pct, "level": level,
                "sustained": sustained, "attention": st.attention if present else None}

    def apply_emotion(self, idx, verdict, emotion, t):
        """Gắn cờ cảm xúc vào kết quả của chỗ `idx` (gọi SAU khi đã có cảm xúc của nhịp này).

        Giữ đúng ý thuật toán gốc: cảm xúc thuộc EMOTION_STRESS_LABELS kéo dài liên tục >=
        EMOTION_STRESS_SUSTAIN_SECONDS thì đặt verdict["emotion_stress"]=True. Đây chỉ là TÍNH HIỆU THAM KHẢO
        (lo lắng khi thi có rất nhiều nguyên nhân bình thường): không cộng điểm, không đổi nhãn, không đổi
        risk_pct. Giống bản gốc: khi cờ bật, luôn thêm dòng "THAM KHAO" vào danh sách vi phạm (chỉ để hiển thị/ghi
        log; risk_pct đã tính xong trước đó nên không bị ảnh hưởng)."""
        if verdict is None:
            return verdict
        st = self.state(idx)
        stress = False
        if emotion in self.stress_labels:
            if st.stress_since is None:
                st.stress_since = t
            stress = (t - st.stress_since) >= self.stress_sustain
        else:
            st.stress_since = None
        verdict["emotion_stress"] = stress
        verdict["emotion_label"] = emotion if stress else ""
        if stress:
            verdict["violations"] = list(verdict["violations"]) + [f"THAM KHAO - CAM XUC LO LANG KEO DAI ({emotion})"]
        return verdict
