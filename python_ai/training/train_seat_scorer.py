r"""
train_seat_scorer.py — Mô hình chấm điểm chỗ ngồi (Logistic Regression trên đặc trưng tổng hợp cửa sổ 10s).

Vì sao không dùng thẳng Transformer: dữ liệu nhãn sạch theo TỪNG CHỖ còn rất ít (vài buổi quay). Với lượng dữ
liệu này, một mô hình tuyến tính trên vài đặc trưng có ý nghĩa vật lý (|yaw|, |dx|, độ dao động...) thường tổng
quát hoá tốt hơn, giải thích được (trọng số = mức đóng góp của từng tín hiệu) và chạy tức thì trên CPU.

Chống rò rỉ / nhãn bẩn:
  * Chỉ dùng các buổi có nhãn THEO TỪNG CHỖ (bỏ buổi gán nhãn cả phòng cùng lúc — xem --exclude).
  * Đánh giá leave-one-session-out: mỗi lần giữ riêng 1 buổi làm tập kiểm tra.

    python python_ai/training/train_seat_scorer.py                 # đánh giá + lưu models/seat_scorer.json
    python python_ai/training/train_seat_scorer.py --no-save       # chỉ đánh giá
"""
import argparse
import json
import os
import sys
import time

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import train_core as TC  # noqa: E402
from eval_ai import auc, event_metrics, rule_scores, window_prf  # noqa: E402
from seat_scorer import AGG_NAMES, aggregate  # noqa: E402

# 29.09.06: 100% thời điểm có nhãn dương gán cho TẤT CẢ chỗ cùng lúc (nhãn mức phòng, không theo chỗ).
# buoi_30_09: nhãn theo chỗ nhưng mọi đặc trưng ~ ngẫu nhiên (AUC~0.5, không cải thiện khi dịch thời gian)
#             -> nghi lệch thứ tự chỗ ngồi giữa lúc gán nhãn và lúc trích đặc trưng. Cần kiểm tra lại video.
DEFAULT_EXCLUDE = ["29.09.06.npz", "buoi_30_09"]

# Các đặc trưng phụ thuộc góc đặt camera / vị trí ghế cố định (dy_last, dy_mean4, dw_abs_last, dh_abs_last, face_frac)
# đảo dấu giữa các video khác nhau -> triệt tiêu để mô hình chỉ học các đại lượng độ lệch hành vi bất biến.
DROP_CONFOUNDED_INDICES = (7, 8, 9, 10, 16)


def fit_logreg(X, y, l2=200.0, iters=600, lr=0.3, pos_weight=1.0, nonneg=True, drop_idx=DROP_CONFOUNDED_INDICES):
    """Logistic regression L2, gradient descent (đủ cho vài nghìn mẫu x ~20 chiều).
    pos_weight=1: KHÔNG cân bằng lớp -> xác suất được hiệu chỉnh (ngưỡng ổn định, không dồn sát 1.0).
    nonneg=True: mọi đặc trưng hoạt động là độ lớn bất thường (|yaw|, |pitch|, |dx|, motion, speed...) nên trọng số
    w >= 0 giúp chống quá khớp (overfitting) khi số lượng video gốc còn ít."""
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    sw = np.where(y == 1, pos_weight, 1.0)
    sw = sw / sw.mean()
    mask = np.ones(d, dtype=np.float64)
    for idx in (drop_idx or ()):
        if 0 <= idx < d:
            mask[idx] = 0.0
    for _ in range(iters):
        z = np.clip(X @ w + b, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        g = (p - y) * sw
        w -= lr * (X.T @ g / n + l2 * w / n)
        if nonneg:
            w = np.maximum(w, 0.0)
        w *= mask
        b -= lr * g.mean()
    return w, b


def predict(model, A):
    z = ((A - model["mean"]) / model["std"]) @ model["w"] + model["b"]
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


POS_WEIGHT = 1.0


def train(A, y, l2, nonneg=True, drop_idx=DROP_CONFOUNDED_INDICES):
    mean, std = A.mean(0), A.std(0) + 1e-6
    w, b = fit_logreg((A - mean) / std, y.astype(float), l2=l2, pos_weight=POS_WEIGHT,
                      nonneg=nonneg, drop_idx=drop_idx)
    return {"mean": mean, "std": std, "w": w, "b": b}


def pick_threshold(y, p, group, seat, t_end, fa_budget):
    """Ngưỡng có recall sự kiện cao nhất mà vẫn giữ báo sai <= fa_budget / giờ / chỗ (trên dữ liệu huấn luyện)."""
    grid = np.unique(np.quantile(p, np.linspace(0.50, 0.999, 80)))   # theo phân vị: không bị kẹt ở trần cố định
    best = (float(grid[-1]), -1.0)
    for thr in grid:
        ev = event_metrics(y, p >= thr, group, seat, t_end)
        if ev["false_alarms_per_seat_hour"] <= fa_budget and ev["event_recall"] > best[1]:
            best = (float(thr), ev["event_recall"])
    return best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data", "labeled"))
    ap.add_argument("--session-map", default=os.path.join(HERE, "session_map_30_09.csv"))
    ap.add_argument("--exclude", nargs="*", default=DEFAULT_EXCLUDE, help="Buổi/file loại khỏi huấn luyện")
    ap.add_argument("--l2", type=float, default=50.0)
    ap.add_argument("--fa-budget", type=float, default=16.0,
                    help="Số báo sai tối đa / giờ / chỗ khi chọn ngưỡng mặc định (medium)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "models", "seat_scorer.json"))
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--pos-weight", type=float, default=1.0, help="Trọng số lớp CHEATING (1 = hiệu chỉnh tự nhiên)")
    ap.add_argument("--no-dedup", action="store_true", help="Tắt khử trùng lặp video gốc (chỉ dùng để tái hiện rò rỉ cũ)")
    a = ap.parse_args()
    global POS_WEIGHT
    POS_WEIGHT = a.pos_weight

    paths = TC.list_labeled(a.data)
    X, y, g, seat, te, used = TC.load_labeled(
        paths, 13, 20, log=print, deduplicate=not a.no_dedup, merge_mode="majority"
    )
    smap = TC.load_session_map(a.session_map) if os.path.exists(a.session_map) else {}
    sess = TC.video_group_keys(g, used, smap) if not a.no_dedup else np.array(
        [smap.get(os.path.basename(used[i]), os.path.basename(used[i])) for i in g]
    )
    keep = ~np.isin(sess, a.exclude)
    X, y, g, seat, te, sess = X[keep], y[keep], g[keep], seat[keep], te[keep], sess[keep]
    # Trong các video có gán nhãn dương (6.10.2026.*, 23_08_6..7), công cụ gán nhãn chỉ lưu nhãn y=1 trên seat=0
    # còn các ghế khác (seat > 0) bị để mặc định y=0 dù thí sinh ở ghế đó cũng cử động/quay đầu (nhãn khuyết).
    # Giữ tất cả ghế ở các video âm hoàn toàn (23_08_1..4) để đo báo sai, và chỉ giữ ghế đã thực sự được gán nhãn ở video dương.
    if not a.no_dedup:
        valid_seat = np.zeros(len(y), bool)
        for s in np.unique(sess):
            m_s = sess == s
            if y[m_s].sum() == 0:
                valid_seat |= m_s
            else:
                for st in np.unique(seat[m_s]):
                    m_st = m_s & (seat == st)
                    if y[m_st].sum() > 0:
                        valid_seat |= m_st
        X, y, g, seat, te, sess = X[valid_seat], y[valid_seat], g[valid_seat], seat[valid_seat], te[valid_seat], sess[valid_seat]
    A = aggregate(X)
    S = rule_scores(X, None)
    rule = S[:, -1] >= 5

    print(f"Dữ liệu nhãn sạch (đã khử trùng lặp video gốc): {len(y)} cửa sổ, {len(np.unique(sess))} video/buổi độc lập, "
          f"CHEATING {y.mean() * 100:.1f}% (loại: {', '.join(a.exclude)})")

    folds = [s for s in np.unique(sess) if y[sess == s].sum() >= 5]
    oof = np.full(len(y), np.nan)
    oof_unconstrained = np.full(len(y), np.nan)
    thr_fold = {}
    for s in folds:
        tr, ts = sess != s, sess == s
        m = train(A[tr], y[tr], a.l2)
        thr_fold[s] = pick_threshold(y[tr], predict(m, A[tr]), g[tr], seat[tr], te[tr], a.fa_budget)
        oof[ts] = predict(m, A[ts])
        m_unc = train(A[tr], y[tr], 5.0, nonneg=False, drop_idx=())
        oof_unconstrained[ts] = predict(m_unc, A[ts])

    ev_mask = np.isin(sess, folds)
    yv, gv, sv, tv = y[ev_mask], g[ev_mask], seat[ev_mask], te[ev_mask]
    pred_lr = np.zeros(len(y), bool)
    for s in folds:
        pred_lr[sess == s] = oof[sess == s] >= thr_fold[s]

    rows = []
    for name, pred, score in [("rule (hiện tại)", rule[ev_mask], S[ev_mask, -4:].mean(1)),
                              ("logreg_cũ (tự do)", oof_unconstrained[ev_mask] >= 0.5, oof_unconstrained[ev_mask]),
                              ("seat_scorer (mới)", pred_lr[ev_mask], oof[ev_mask])]:
        p, r, f = window_prf(yv, pred)
        ev = event_metrics(yv, pred, gv, sv, tv)
        rows.append((name, p, r, f, auc(yv, score), ev))
    print("\n=== LEAVE-ONE-VIDEO-OUT (Không rò rỉ dữ liệu giữa các file trùng X) ===")
    print(f"{'Phương pháp':<18}{'P':>6}{'R':>6}{'F1':>6}{'AUC':>6}{'Recall SK':>11}{'Prec SK':>9}{'Báo sai/giờ/chỗ':>17}")
    for name, p, r, f, au, ev in rows:
        print(f"{name:<18}{p:>6.2f}{r:>6.2f}{f:>6.2f}{au:>6.2f}{ev['event_recall']:>11.2f}{ev['event_precision']:>9.2f}"
              f"{ev['false_alarms_per_seat_hour']:>17.1f}")
    # Đường đánh đổi: cùng 1 mức báo sai, phương pháp nào bắt được nhiều sự kiện hơn?
    print("\nĐánh đổi recall sự kiện <-> báo sai/giờ/chỗ (ngưỡng chọn trên buổi huấn luyện của từng fold):")
    print(f"  {'ngân sách':>10} | {'seat_scorer: recall / báo sai':>32} | {'luật k-of-n gần nhất: recall / báo sai':>40}")
    rule_variants = {f"{k}of{n}": (S[:, -n:] >= 5).sum(1) >= k for k, n in [(1, 1), (2, 3), (3, 4), (4, 6), (5, 8)]}
    rule_ev = {name: event_metrics(yv, m[ev_mask], gv, sv, tv) for name, m in rule_variants.items()}
    for budget in [1.0, 2.0, 4.0, 8.0, 16.0]:
        pred_b = np.zeros(len(y), bool)
        for s in folds:
            tr = sess != s
            m_tr = train(A[tr], y[tr], a.l2)
            t_b = pick_threshold(y[tr], predict(m_tr, A[tr]), g[tr], seat[tr], te[tr], budget)
            pred_b[sess == s] = oof[sess == s] >= t_b
        ev = event_metrics(yv, pred_b[ev_mask], gv, sv, tv)
        ok = {k: v for k, v in rule_ev.items() if v["false_alarms_per_seat_hour"] <= max(budget, ev["false_alarms_per_seat_hour"])}
        best_rule = max(ok.items(), key=lambda kv: kv[1]["event_recall"]) if ok else None
        rtxt = (f"{best_rule[0]}: {best_rule[1]['event_recall']:.2f} / {best_rule[1]['false_alarms_per_seat_hour']:.1f}"
                if best_rule else "không đạt")
        print(f"  {budget:>10.0f} | {ev['event_recall']:>21.2f} / {ev['false_alarms_per_seat_hour']:<8.1f} | {rtxt:>40}")

    print("\nTheo từng buổi (seat_scorer): AUC | recall SK | báo sai/giờ/chỗ")
    for s in folds:
        m = sess == s
        ev = event_metrics(y[m], pred_lr[m], g[m], seat[m], te[m])
        print(f"  {s:<20} {auc(y[m], oof[m]):.2f} | {ev['event_recall']:.2f} | {ev['false_alarms_per_seat_hour']:.1f}"
              f"   (ngưỡng {thr_fold[s]:.3f})")

    # Mô hình cuối: huấn luyện trên toàn bộ dữ liệu sạch
    final = train(A, y, a.l2)
    p_all = predict(final, A)
    thr = pick_threshold(y, p_all, g, seat, te, a.fa_budget)
    # 3 mức độ nhạy cho lúc chạy thật (Config.SEAT_SCORER_SENSITIVITY / --sensitivity)
    sensitivity = {name: pick_threshold(y, p_all, g, seat, te, budget)
                   for name, budget in [("low", 8.0), ("medium", 16.0), ("high", 28.0)]}
    print(f"Ngưỡng theo độ nhạy (báo sai/giờ/chỗ trên dữ liệu huấn luyện <= 8 / 16 / 28): {sensitivity}")
    order = np.argsort(-np.abs(final["w"]))
    print("\nTrọng số (đã chuẩn hoá) — tín hiệu đóng góp nhiều nhất:")
    for i in order[:8]:
        print(f"  {AGG_NAMES[i]:<16} {final['w'][i]:+.3f}")
    print(f"Ngưỡng cuối (báo sai <= {a.fa_budget}/giờ/chỗ trên dữ liệu huấn luyện): {thr:.3f}")

    if not a.no_save:
        out = {"type": "logreg_v1", "features": AGG_NAMES, "mean": final["mean"].tolist(),
               "std": final["std"].tolist(), "w": final["w"].tolist(), "b": float(final["b"]),
               "threshold": thr, "thresholds": sensitivity, "seq_len": 20, "feature_dim": 13, "feature_hz": 2.0,
               "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), "excluded": a.exclude,
               "n_windows": int(len(y)), "loso": {name: {"win_f1": f, "auc": au, **ev}
                                                  for name, p, r, f, au, ev in rows}}
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)
        print(f"Đã lưu {a.out}")


if __name__ == "__main__":
    main()
