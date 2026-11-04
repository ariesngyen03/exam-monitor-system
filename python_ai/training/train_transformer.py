r"""
train_transformer.py — Huấn luyện lại Transformer từ dữ liệu bạn đã gán nhãn bằng label_tool.py.

    python train_transformer.py                       # đọc train_data\labeled\*.npz
    python train_transformer.py --data train_data\labeled --epochs 80

Kết quả: models\transformer_seat.pt + models\transformer_seat_meta.json (số chiều, độ dài chuỗi, tần số lấy mẫu,
ngưỡng chọn theo F1 trên tập kiểm tra). giamsatthi.py tự đọc meta này và từ chối nạp model nếu cấu hình không khớp.
LƯU Ý: script này chưa được chạy thử trong môi trường phát triển (không có PyTorch ở đó); phần chia dữ liệu/chọn ngưỡng
(train_core.py) đã có kiểm thử. Nếu gặp lỗi, gửi lại nguyên văn thông báo.
"""
import argparse
import copy
import json
import os
import time

import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # python_ai/ (ai_core)
sys.path.insert(0, HERE)

import train_core as TC  # noqa: E402
from ai_core import CheatingTransformer, Config  # noqa: E402


def predict(model, X, device, bs=512):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).to(device)
            out.append(torch.softmax(model(xb), dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data", "labeled"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--split", choices=["video", "block"], default="video",
                     help="video (mac dinh, khuyen dung): giu nguyen ca video cho tap kiem tra, khong ro ri "
                          "nguoi/nen. block: cat nho trong cung video, dung duoc nhieu du lieu huan luyen hon "
                          "nhung so lieu kiem tra lac quan hon thuc te.")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--session-map", default=None,
                    help="File CSV 'ten_file.npz,ma_buoi' — BẮT BUỘC khai báo nếu nhiều file .npz là các "
                         "đoạn CẮT RA từ CÙNG 1 buổi quay liên tục (cùng 1 nhóm người suốt từ đầu đến "
                         "cuối), nếu không --split video sẽ RÒ RỈ NGƯỜI giữa các đoạn của cùng 1 buổi.")
    a = ap.parse_args()

    paths = TC.list_labeled(a.data)
    X, y, g, seat, te, used = TC.load_labeled(paths, Config.FEATURE_DIM, Config.SEQ_LEN)
    print(f"Dữ liệu: {len(used)} file, {len(y)} cửa sổ ({Config.SEQ_LEN} x {Config.FEATURE_DIM}), CHEATING {int(y.sum())} ({y.mean() * 100:.1f}%)")
    if int(y.sum()) < 20:
        print("[WARN] Rất ít mẫu CHEATING (<20): model học được rất ít, kết quả kiểm tra không đáng tin. Nên gán nhãn thêm.")
    session_map = TC.load_session_map(a.session_map) if a.session_map else None
    if a.split == "video":
        if session_map is None:
            print("[WARN] Không có --session-map: nếu nhiều file .npz là đoạn cắt từ CÙNG 1 buổi quay "
                 "liên tục (cùng 1 nhóm người), --split video có thể RÒ RỈ NGƯỜI giữa train/val, cho số "
                 "liệu lạc quan giả — xem lại tên file, nếu đúng vậy hãy tạo file session-map rồi chạy lại.")
        val = TC.video_split(y, g, a.val_frac, a.seed, used=used, session_map=session_map)
        if len(np.unique(g)) < 5:
            print(f"[WARN] Chi co {len(np.unique(g))} video: chia theo video se co phuong sai cao (ket qua phu "
                  f"thuoc nhieu vao video nao roi vao tap kiem tra). So lieu duoi day mang tinh tham khao.")
    else:
        val = TC.block_split(y, g, seat, te, a.val_frac, a.seed)
        print("[WARN] Dang dung --split block: cung nguoi/nen co the vua o tap huan luyen vua o tap kiem tra "
              "(ro ri), so lieu kiem tra co the lac quan hon thuc te khi gap nguoi moi.")
    Xtr, ytr, Xv, yv = X[~val], y[~val], X[val], y[val]
    print(f"Huấn luyện: {len(ytr)} (CHEATING {int(ytr.sum())}) | Kiểm tra: {len(yv)} (CHEATING {int(yv.sum())})")
    val_files = sorted({os.path.basename(used[gi]) for gi in np.unique(g[val])})
    print(f"Các file THỰC SỰ nằm trong tập kiểm tra (không dùng để huấn luyện vòng này): {', '.join(val_files)}")

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CheatingTransformer(input_dim=Config.FEATURE_DIM).to(device)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(TC.class_weights(ytr)).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)), batch_size=a.batch, shuffle=True)

    best_score, best_state, best_ep, bad = -1e9, None, 0, 0
    for ep in range(1, a.epochs + 1):
        model.train()
        tl = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            xb = xb + 0.02 * torch.randn_like(xb)                    # nhiễu nhẹ chống học vẹt
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * len(xb)
        pv = predict(model, Xv, device)
        vloss = float(-np.mean(np.log(np.where(yv == 1, pv, 1 - pv) + 1e-7))) if len(yv) else 0.0
        thr, f1 = TC.best_threshold(yv, pv) if len(yv) else (0.6, None)
        score = f1 if f1 is not None else -vloss
        sched.step(vloss)
        print(f"Epoch {ep:03d} | loss huấn luyện {tl / len(ytr):.4f} | loss kiểm tra {vloss:.4f} | F1 kiểm tra {f1 if f1 is not None else float('nan'):.3f} (ngưỡng {thr:.2f})")
        if score > best_score:
            best_score, best_state, best_ep, bad = score, copy.deepcopy(model.state_dict()), ep, 0
        else:
            bad += 1
            if bad >= a.patience:
                print(f"Dừng sớm (không cải thiện sau {a.patience} epoch). Tốt nhất ở epoch {best_ep}.")
                break

    model.load_state_dict(best_state)
    pv = predict(model, Xv, device)
    thr, f1 = TC.best_threshold(yv, pv, default=Config.CHEAT_THRESHOLD)
    p, r, f = TC.prf(yv, pv, thr)
    os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)
    torch.save(best_state, Config.MODEL_PATH)
    meta = {"feature_dim": Config.FEATURE_DIM, "seq_len": Config.SEQ_LEN, "feature_hz": Config.FEATURE_HZ,
            "threshold": thr, "val_precision": p, "val_recall": r, "val_f1": f, "best_epoch": best_ep,
            "n_train": int(len(ytr)), "n_val": int(len(yv)), "pos_train": int(ytr.sum()), "pos_val": int(yv.sum()),
            "files": [str(u) for u in used], "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "split": a.split, "session_map_used": a.session_map or None}
    with open(Config.MODEL_META_PATH, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu {Config.MODEL_PATH}\nKiểm tra: precision {p:.2f} | recall {r:.2f} | F1 {f:.2f} | ngưỡng {thr:.2f}")
    if int(yv.sum()) < 10:
        print("[WARN] Tập kiểm tra có <10 mẫu CHEATING: các con số trên chỉ mang tính tham khảo.")


if __name__ == "__main__":
    main()