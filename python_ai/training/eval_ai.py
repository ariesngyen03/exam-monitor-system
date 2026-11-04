r"""
eval_ai.py — Đánh giá AI THEO SỰ KIỆN trên dữ liệu đã gán nhãn, so sánh các cách ra quyết định.

Vì sao theo sự kiện: giám thị không quan tâm "bao nhiêu cửa sổ 10s đúng", mà quan tâm
  (1) bao nhiêu lần gian lận thật được báo (event recall), và
  (2) bao nhiêu cảnh báo SAI mỗi giờ họ phải xử lý (false alarms / giờ).

Các phương pháp so sánh (cùng 1 tập kiểm tra, tái tạo đúng cách chia của train_transformer.py):
  rule        : luật hành vi (behavior.rule_evaluate) tại mẫu cuối cửa sổ — đúng như hệ thống đang chạy
  rule_k      : luật + bền vững — điểm luật >= 5 ở >= k trong n mẫu gần nhất (lọc dao động tức thời)
  tf          : Transformer (prob >= ngưỡng trong meta)
  and / or    : hợp nhất luật & Transformer (HYBRID_FUSION_MODE)

    python python_ai/training/eval_ai.py
    python python_ai/training/eval_ai.py --model python_ai/models/transformer_seat.pt --all-rules
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl_exam_monitor"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import train_core as TC  # noqa: E402
from behavior import rule_evaluate  # noqa: E402
from seat_scorer import SeatScorer, aggregate  # noqa: E402

FEATURE_DIM, SEQ_LEN, HZ = 13, 20, 2.0
STRIDE_SEC = 1.0          # make_windows(stride=2) ở 2 Hz -> 1 cửa sổ / giây
HOLD_SEC = 5.0            # BehaviorEngine giữ CHEATING tối thiểu SEAT_EXIT_SECONDS -> gộp các lần chớp tắt
TOL_SEC = 2.0             # dung sai khi ghép sự kiện dự đoán với sự kiện thật


# ── giải chuẩn hoá vector 13 chiều (features.to_vector) về đại lượng thô cho luật ─────────────
def raw_from_vec(v):
    absent = v[9] <= -0.999 and v[10] <= -0.999
    return absent, {"face_ok": float(v[0]), "yaw": float(v[1]) * 90.0, "pitch": float(v[2]) * 45.0,
                    "roll": float(v[3]) * 45.0, "ear_l": float(v[4]) * 0.4, "ear_r": float(v[5]) * 0.4,
                    "gaze": float(v[6]), "dx": float(v[7]) * 2.0, "dy": float(v[8]) * 2.0,
                    "dw": float(v[9]), "dh": float(v[10]), "speed": float(v[11]), "motion": float(v[12])}


def rule_scores(X, cfg):
    """Điểm luật cho TỪNG mẫu của từng cửa sổ: (N, SEQ_LEN)."""
    S = np.zeros(X.shape[:2], np.float32)
    for i in range(X.shape[0]):
        for t in range(X.shape[1]):
            absent, raw = raw_from_vec(X[i, t])
            S[i, t] = 0.0 if absent else rule_evaluate(raw, cfg)[0]
    return S


def tf_probs(model_path, X):
    import torch
    from ai_core import CheatingTransformer
    model = CheatingTransformer(input_dim=FEATURE_DIM)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 1024):
            out.append(torch.softmax(model(torch.from_numpy(X[i:i + 1024])), dim=-1)[:, 1].numpy())
    return np.concatenate(out)


def auc(y, s):
    """ROC AUC theo hạng (Mann-Whitney), không cần sklearn."""
    y = np.asarray(y).astype(bool)
    if y.all() or (~y).all():
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    sorted_s = s[order]
    i = 0
    while i < len(s):                                   # hạng trung bình cho giá trị bằng nhau
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    n1, n0 = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def runs(mask, t):
    """Các đoạn liên tiếp mask=True trên trục thời gian t (giây) -> [(t_bắt_đầu, t_kết_thúc)]."""
    ev, start, prev = [], None, None
    for m, ti in zip(mask, t):
        if m and start is None:
            start = ti
        if start is not None and (not m or (prev is not None and ti - prev > 3 * STRIDE_SEC)):
            ev.append((start, prev))
            start = ti if m else None
        prev = ti
    if start is not None:
        ev.append((start, prev))
    return ev


def hold(mask, t, sec):
    """Mô phỏng chống nhiễu của BehaviorEngine: đã báo thì giữ thêm `sec` giây."""
    out, until = np.zeros_like(mask), -1e9
    for i, (m, ti) in enumerate(zip(mask, t)):
        if m:
            until = ti + sec
        out[i] = m or ti <= until
    return out


def event_metrics(y, pred, group, seat, t_end):
    tp_true = n_true = tp_pred = n_pred = 0
    seconds = 0.0
    for key in sorted(set(zip(group.tolist(), seat.tolist()))):
        idx = np.where((group == key[0]) & (seat == key[1]))[0]
        idx = idx[np.argsort(t_end[idx])]
        t = t_end[idx]
        seconds += len(idx) * STRIDE_SEC
        true_ev = runs(y[idx] == 1, t)
        pred_ev = runs(hold(pred[idx], t, HOLD_SEC), t)
        hit = lambda a, b: a[0] - TOL_SEC <= b[1] and b[0] - TOL_SEC <= a[1]
        n_true += len(true_ev)
        tp_true += sum(any(hit(te, pe) for pe in pred_ev) for te in true_ev)
        n_pred += len(pred_ev)
        tp_pred += sum(any(hit(pe, te) for te in true_ev) for pe in pred_ev)
    hours = seconds / 3600.0
    fa = n_pred - tp_pred
    return {"event_recall": tp_true / n_true if n_true else float("nan"),
            "event_precision": tp_pred / n_pred if n_pred else float("nan"),
            "false_alarms_per_seat_hour": fa / hours if hours else float("nan"),
            "true_events": n_true, "pred_events": n_pred, "seat_hours": hours}


def window_prf(y, pred):
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def report(name, y, pred, score, group, seat, t_end):
    p, r, f = window_prf(y, pred)
    ev = event_metrics(y, pred, group, seat, t_end)
    return {"method": name, "win_precision": p, "win_recall": r, "win_f1": f,
            "auc": auc(y, score) if score is not None else float("nan"), **ev}


def print_table(rows, title):
    print(f"\n=== {title} ===")
    hdr = f"{'Phương pháp':<16}{'P(cửa sổ)':>10}{'R(cửa sổ)':>10}{'F1':>7}{'AUC':>7}{'Recall SK':>11}{'Prec SK':>9}{'Báo sai/giờ/chỗ':>17}{'SK dự đoán':>12}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['method']:<16}{r['win_precision']:>10.2f}{r['win_recall']:>10.2f}{r['win_f1']:>7.2f}{r['auc']:>7.2f}"
              f"{r['event_recall']:>11.2f}{r['event_precision']:>9.2f}{r['false_alarms_per_seat_hour']:>17.1f}{r['pred_events']:>12d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data", "labeled"))
    ap.add_argument("--session-map", default=os.path.join(HERE, "session_map_30_09.csv"))
    ap.add_argument("--model", default=os.path.join(os.path.dirname(HERE), "models", "transformer_seat.pt"))
    ap.add_argument("--meta", default=os.path.join(os.path.dirname(HERE), "models", "transformer_seat_meta.json"))
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all-rules", action="store_true", help="Đánh giá thêm luật trên TOÀN BỘ dữ liệu (luật không huấn luyện)")
    ap.add_argument("--dedup", action="store_true", help="Khử trùng lặp các file .npz có cùng ma trận đặc trưng X")
    ap.add_argument("--json", default=None, help="Ghi kết quả ra file JSON")
    a = ap.parse_args()

    paths = TC.list_labeled(a.data)
    X, y, g, seat, te, used = TC.load_labeled(paths, FEATURE_DIM, SEQ_LEN, log=lambda *_: None, deduplicate=a.dedup)
    smap = TC.load_session_map(a.session_map) if a.session_map and os.path.exists(a.session_map) else None
    val = TC.video_split(y, g, a.val_frac, a.seed, used=used, session_map=smap)
    thr = 0.60
    if os.path.exists(a.meta):
        with open(a.meta, encoding="utf-8") as f:
            thr = float(json.load(f).get("threshold", thr))

    print(f"Dữ liệu: {len(used)} video/file, {len(y)} cửa sổ, CHEATING {y.mean() * 100:.1f}%")
    print(f"Tập kiểm tra (theo buổi quay): {int(val.sum())} cửa sổ, CHEATING {y[val].mean() * 100:.1f}% | "
          f"file: {', '.join(sorted({os.path.basename(used[i]) for i in np.unique(g[val])}))}")
    print(f"Ngưỡng Transformer (meta): {thr:.2f} | gộp sự kiện: giữ {HOLD_SEC:.0f}s, dung sai {TOL_SEC:.0f}s")

    S = rule_scores(X, None)
    last = S[:, -1]
    rule = last >= 5
    rule_k = {f"rule_{k}of{n}": (S[:, -n:] >= 5).sum(1) >= k for k, n in [(2, 3), (3, 4), (4, 6)]}
    rule_cont = S[:, -4:].mean(1)

    rows_val, results = [], {}
    yv, gv, sv, tv = y[val], g[val], seat[val], te[val]
    rows_val.append(report("rule", yv, rule[val], rule_cont[val], gv, sv, tv))
    for name, m in rule_k.items():
        rows_val.append(report(name, yv, m[val], rule_cont[val], gv, sv, tv))
    scorer = SeatScorer.load_default(os.path.join(os.path.dirname(HERE), "models"))
    if scorer is not None:
        A_val = aggregate(X[val])
        z_val = ((A_val - scorer.mean) / scorer.std) @ scorer.w + scorer.b
        p_sc = 1.0 / (1.0 + np.exp(-np.clip(z_val, -30, 30)))
        rows_val.append(report("seat_scorer", yv, p_sc >= scorer.threshold, p_sc, gv, sv, tv))
    if os.path.exists(a.model):
        P = tf_probs(a.model, X[val])
        tf = P >= thr
        rows_val.append(report("tf", yv, tf, P, gv, sv, tv))
        rows_val.append(report("and (hiện tại)", yv, tf & rule[val], np.minimum(P, rule_cont[val] / 10), gv, sv, tv))
        rows_val.append(report("or", yv, tf | rule[val], np.maximum(P, rule_cont[val] / 10), gv, sv, tv))
        for name, m in rule_k.items():
            rows_val.append(report(f"and+{name[5:]}", yv, tf & m[val], None, gv, sv, tv))
    print_table(rows_val, "TẬP KIỂM TRA (buổi quay chưa dùng để huấn luyện)")
    results["val"] = rows_val

    if a.all_rules:
        rows_all = [report("rule", y, rule, rule_cont, g, seat, te)]
        for name, m in rule_k.items():
            rows_all.append(report(name, y, m, rule_cont, g, seat, te))
        if scorer is not None:
            A_all = aggregate(X)
            z_all = ((A_all - scorer.mean) / scorer.std) @ scorer.w + scorer.b
            p_all = 1.0 / (1.0 + np.exp(-np.clip(z_all, -30, 30)))
            rows_all.append(report("seat_scorer", y, p_all >= scorer.threshold, p_all, g, seat, te))
        print_table(rows_all, "LUẬT TRÊN TOÀN BỘ DỮ LIỆU (luật không học từ dữ liệu)")
        results["all_rules"] = rows_all

    print("\nGhi chú: 'Báo sai/giờ/chỗ' nhân với số chỗ ngồi trong phòng = số cảnh báo sai giám thị nhận mỗi giờ.")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Đã ghi {a.json}")


if __name__ == "__main__":
    main()
