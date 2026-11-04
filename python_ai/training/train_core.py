"""train_core.py — Phần huấn luyện KHÔNG cần torch (nạp dữ liệu, khử trùng lặp video gốc, chia train/val, chọn ngưỡng): kiểm thử được."""
import glob
import hashlib
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def x_hash(X):
    """Mã băm MD5 (8 ký tự) của tensor đặc trưng X để nhận diện các file .npz xuất phát từ CÙNG 1 video gốc."""
    return hashlib.md5(np.ascontiguousarray(X).tobytes()).hexdigest()[:8]


def load_labeled(paths, feature_dim, seq_len, log=print, deduplicate=False, merge_mode="majority"):
    """Gộp các file train_data/labeled/*.npz. Bỏ qua file khác số chiều/độ dài (VD dữ liệu cũ 22 chiều).

    Nếu `deduplicate=True`: các file .npz có CÙNG ma trận đặc trưng X (cùng 1 video gốc được gán nhãn nhiều lượt)
    sẽ được gộp thành 1 video duy nhất, nhãn `y` được hợp nhất theo `merge_mode` ('majority' = đa số >= 0.5,
    'union' = phép OR) để khử hoàn toàn rò rỉ dữ liệu giữa tập huấn luyện và tập kiểm tra.
    """
    raw_items = []
    for p in paths:
        z = np.load(p, allow_pickle=False)
        if "X" not in z.files or "y" not in z.files:
            log(f"[BỎ QUA] {os.path.basename(p)}: không phải file gán nhãn hợp lệ")
            continue
        X = z["X"]
        if X.ndim != 3 or X.shape[2] != feature_dim or X.shape[1] != seq_len:
            log(f"[BỎ QUA] {os.path.basename(p)}: kích thước {X.shape} không khớp (cần (*, {seq_len}, {feature_dim}))")
            continue
        n = len(X)
        X_f = X.astype(np.float32)
        y_i = z["y"].astype(np.int64)
        s_i = z["seat"].astype(np.int32) if "seat" in z.files else np.zeros(n, np.int32)
        t_f = z["t_end"].astype(np.float32) if "t_end" in z.files else np.arange(n, dtype=np.float32)
        raw_items.append((p, X_f, y_i, s_i, t_f))
    if not raw_items:
        raise RuntimeError("Không có file gán nhãn hợp lệ nào.")

    if not deduplicate:
        Xs, ys, grp, seat, te, used = [], [], [], [], [], []
        for p, X_f, y_i, s_i, t_f in raw_items:
            Xs.append(X_f)
            ys.append(y_i)
            grp.append(np.full(len(X_f), len(used), np.int32))
            seat.append(s_i)
            te.append(t_f)
            used.append(p)
        return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(grp),
                np.concatenate(seat), np.concatenate(te), used)

    by_vid = {}
    for item in raw_items:
        h = x_hash(item[1])
        by_vid.setdefault(h, []).append(item)

    Xs, ys, grp, seat, te, used = [], [], [], [], [], []
    for h, items in by_vid.items():
        p0, X0, _, s0, t0 = items[0]
        if len(items) == 1:
            y_merged = items[0][2]
        else:
            y_stack = np.stack([it[2] for it in items])
            if merge_mode == "union":
                y_merged = y_stack.max(axis=0).astype(np.int64)
            else:
                y_merged = (y_stack.mean(axis=0) >= 0.5).astype(np.int64)
            names = [os.path.basename(it[0]) for it in items]
            log(f"[GỘP VIDEO TRÙNG X] {names[0]} (+{len(items) - 1} file trùng X) -> {len(X0)} cửa sổ, "
                f"dương={int(y_merged.sum())}")
        Xs.append(X0)
        ys.append(y_merged)
        grp.append(np.full(len(X0), len(used), np.int32))
        seat.append(s0)
        te.append(t0)
        used.append(p0)
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(grp),
            np.concatenate(seat), np.concatenate(te), used)


def video_group_keys(group, used, session_map=None):
    """Ánh xạ mỗi mẫu trong `group` về mã video/buổi quay thực sự (gộp cả theo `session_map` LẪN theo mã băm
    nội dung `X` của file gốc nếu nhiều file .npz có cùng ma trận đặc trưng `X`)."""
    hash_to_canonical = {}
    group_to_key = {}
    for gi in np.unique(group):
        p = used[int(gi)]
        base = os.path.basename(p)
        mapped = session_map.get(base, base) if session_map else base
        if os.path.exists(p):
            try:
                z = np.load(p, allow_pickle=False)
                if "X" in z.files:
                    h = x_hash(z["X"])
                    mapped = hash_to_canonical.setdefault(h, mapped)
            except Exception:
                pass
        group_to_key[int(gi)] = mapped
    return np.array([group_to_key[int(gi)] for gi in group])


def block_split(y, group, seat, t_end, val_frac=0.2, seed=0, block=20):
    """Chia theo KHỐI cửa sổ liên tiếp của cùng (video, chỗ) để giảm rò rỉ (các cửa sổ chồng lấn rất giống nhau).
    Trả về mask val (bool). Cố gắng để cả train và val đều có mẫu CHEATING nếu tổng số đủ."""
    n = len(y)
    block_id = np.zeros(n, np.int64)
    key = {}
    for i in np.lexsort((t_end, seat, group)):
        k = (int(group[i]), int(seat[i]))
        rank = key.get(k, 0)
        key[k] = rank + 1
        block_id[i] = hash((k, rank // block)) & 0x7FFFFFFF
    blocks = np.unique(block_id)
    rng = np.random.RandomState(seed)
    rng.shuffle(blocks)
    val = np.zeros(n, bool)
    for b in blocks:
        if val.sum() >= val_frac * n:
            break
        val |= block_id == b
    pos_total = int(y.sum())
    if pos_total >= 4:
        def swap_in_positive_block():
            cand = [b for b in blocks if (y[(block_id == b)] == 1).any() and not val[block_id == b].any()]
            if cand:
                val[block_id == cand[0]] = True
        if int(y[val].sum()) == 0:
            swap_in_positive_block()
        if int(y[~val].sum()) == 0:            # val ăn hết mẫu dương -> trả lại 1 khối
            for b in blocks:
                if val[block_id == b].any() and (y[block_id == b] == 1).any():
                    val[block_id == b] = False
                    break
    return val


def load_session_map(path):
    """Đọc file ánh xạ TÊN FILE -> MÃ BUỔI QUAY, mỗi dòng 'ten_file.npz,ma_buoi' (dòng trống/bắt đầu bằng
    '#' bị bỏ qua). Dùng khi nhiều file .npz thực ra là các đoạn CẮT RA từ CÙNG 1 buổi quay (cùng 1 nhóm
    người) — nếu không khai báo, mỗi file bị coi là 1 buổi độc lập, có thể gây RÒ RỈ NGƯỜI nếu thực tế
    không phải vậy (xem video_split)."""
    m = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, _, session = line.partition(",")
            if not session:
                raise ValueError(f"Dòng sai định dạng trong {path} (cần 'ten_file,ma_buoi'): {line!r}")
            m[name.strip()] = session.strip()
    return m


def video_split(y, group, val_frac=0.2, seed=0, used=None, session_map=None):
    """Chia theo TỪNG BUỔI QUAY nguyên vẹn (không cắt nhỏ) — không rò rỉ người/nền giữa train và kiểm
    tra, nên số liệu phản ánh đúng hơn khả năng nhận ra NGƯỜI MỚI.

    QUAN TRỌNG: đơn vị chia là BUỔI QUAY, không phải FILE. Nếu nhiều file .npz là các đoạn cắt ra từ
    CÙNG 1 buổi quay liên tục (cùng 1 nhóm người suốt từ đầu đến cuối), truyền `used` (danh sách đường
    dẫn file, cùng thứ tự với chỉ số trong `group`) và `session_map` (từ load_session_map) để các đoạn
    đó LUÔN cùng về 1 phía train/val — nếu không, các đoạn của cùng 1 buổi có thể bị tách đôi, khiến
    model "đã thấy" những người xuất hiện trong phần val ngay trong lúc huấn luyện (rò rỉ người), cho
    điểm cao giả tạo dù model không hề tổng quát hoá được sang buổi quay mới.

    Không truyền session_map -> mỗi FILE là 1 buổi độc lập (hành vi cũ, giữ tương thích ngược)."""
    if used is not None:
        session_key = video_group_keys(group, used, session_map)
    elif session_map:
        session_of_group = {gi: session_map.get(os.path.basename(used[gi]), os.path.basename(used[gi]))
                            for gi in np.unique(group)}
        session_key = np.array([session_of_group[gi] for gi in group])
    else:
        session_key = group   # mỗi file = 1 buổi (hành vi cũ)
    sessions = np.unique(session_key)
    rng = np.random.RandomState(seed)
    rng.shuffle(sessions)
    n = len(y)
    val = np.zeros(n, bool)
    for s in sessions:
        if val.sum() >= val_frac * n:
            break
        val |= session_key == s
    pos_total = int(y.sum())
    if pos_total >= 4:
        if int(y[val].sum()) == 0:                          # val chưa có mẫu CHEATING nào -> đổi 1 buổi có
            for s in sessions:
                if not val[session_key == s].any() and (y[session_key == s] == 1).any():
                    val |= session_key == s
                    break
        if int(y[~val].sum()) == 0:                          # val ăn hết mẫu CHEATING -> trả lại 1 buổi
            for s in sessions:
                if val[session_key == s].any() and (y[session_key == s] == 1).any():
                    val &= session_key != s
                    break
    return val


def prf(y, prob, thr):
    pred = prob >= thr
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def best_threshold(y, prob, default=0.6, grid=None):
    """Ngưỡng tối đa hoá F1 trên tập kiểm tra. Không có mẫu CHEATING -> giữ mặc định."""
    if int(y.sum()) == 0:
        return default, None
    best = (-1.0, default)
    for t in (grid if grid is not None else np.arange(0.30, 0.96, 0.05)):
        f = prf(y, prob, t)[2]
        if f >= best[0]:                       # hoà thì lấy ngưỡng cao hơn (ít báo động giả hơn)
            best = (f, float(t))
    return best[1], best[0]


def class_weights(y, cap=10.0):
    pos = max(int(y.sum()), 1)
    neg = max(int((y == 0).sum()), 1)
    return np.array([1.0, min(neg / pos, cap)], np.float32)


def list_labeled(folder):
    return sorted(glob.glob(os.path.join(folder, "*.npz")))