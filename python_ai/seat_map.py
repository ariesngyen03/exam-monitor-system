"""
seat_map.py — Bản đồ chỗ ngồi cho camera đặt trên cao quan sát cả lớp.

Ý tưởng: danh tính của một thí sinh = CHỖ NGỒI của họ, không phải track ID.
  1. Giai đoạn học (vài giây đầu): gom các lần dò người thành cụm, mỗi cụm là 1 chỗ.
  2. Khoá bản đồ, đánh nhãn "Dr-Gc" (VD "D1-G2" = Dãy 1 - Ghế 2).
  3. Sau đó mỗi lần dò người được ghép vào chỗ gần nhất bằng thuật toán Hungarian
     (1 chỗ - 1 người), nên nhãn vị trí luôn ổn định dù mất/có mặt hay dò nhấp nháy.

Module không import ai_core: nhận cfg bất kỳ (getattr có mặc định) nên kiểm thử độc lập được.
"""
import math
import statistics

import numpy as np
from scipy.optimize import linear_sum_assignment


def anchor_of(bbox, head_ratio=0.30):
    """Điểm neo của 1 người: giữa theo chiều ngang, ~30% từ đỉnh box xuống (vùng đầu/vai)."""
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2.0, y0 + head_ratio * (y1 - y0))


def _seat_extent(s):
    """Khung của 1 chỗ theo điểm neo (cùng công thức với giamsatthi._seat_box)."""
    return (s["ax"] - s["w"] / 2, s["ay"] - 0.3 * s["h"], s["ax"] + s["w"] / 2, s["ay"] + 0.7 * s["h"])


def _drop_merged(seats):
    """Bỏ cụm "gộp người": bộ dò thỉnh thoảng khoanh 2 người ngồi trước-sau thành 1 box lớn, box đó đủ ổn định
    để thành 1 chỗ giả chồng lên 2 chỗ thật. Nhận biết: khung chứa điểm neo của >= 2 chỗ khác."""
    if len(seats) < 3:
        return seats
    out = []
    for s in seats:
        x0, y0, x1, y1 = _seat_extent(s)
        inside = sum(1 for o in seats if o is not s and x0 <= o["ax"] <= x1 and y0 <= o["ay"] <= y1)
        if inside < 2:
            out.append(s)
    return out if out else seats


class SeatMap:
    def __init__(self, cfg=None):
        g = (lambda k, d: getattr(cfg, k, d)) if cfg is not None else (lambda k, d: d)
        self.calib_seconds = g("SEAT_CALIBRATION_SECONDS", 15.0)
        self.min_hit_ratio = g("SEAT_MIN_HIT_RATIO", 0.35)
        self.cluster_radius_w = g("SEAT_CLUSTER_RADIUS_W", 0.6)
        self.match_radius_w = g("SEAT_MATCH_RADIUS_W", 1.5)   # phải lớn hơn SEAT_LEAN_STRONG, nếu không luật nghiêng người không bao giờ kích hoạt
        self.row_gap_h = g("SEAT_ROW_GAP_H", 0.6)
        self.head_ratio = g("SEAT_HEAD_RATIO", 0.30)
        self.min_updates = g("SEAT_MIN_UPDATES", 3)
        self.grid_complete = g("SEAT_GRID_COMPLETE", True)
        # Quán tính khi ghép người-vào-chỗ: ưu tiên GIỮ NGUYÊN ghép của nhịp trước nếu người vẫn còn gần
        # đó, thay vì tính lại hoàn toàn độc lập mỗi nhịp — tránh 2 chỗ gần nhau bị đổi qua lại do dao
        # động tự nhiên nhỏ (lắc người, hơi nghiêng...) đẩy "chỗ tối ưu" lật từ chỗ này sang chỗ kia dù
        # thực ra vẫn cùng 1 người ngồi yên. Khi đổi qua lại, TOÀN BỘ trạng thái theo dõi của chỗ đó
        # (CHEATING/NORMAL, cảm xúc...) bị "chuyển nhà" theo, trông như reset/nhấp nháy.
        self.sticky_discount = g("SEAT_STICKY_DISCOUNT", 0.7)
        self._last_anchor = {}
        self.col_right_to_left = g("SEAT_COL_RIGHT_TO_LEFT", False)  # True: đánh "Cho 1" từ bên PHẢI khung hình
        # Phòng có bàn xếp thành DÃY KÉO DÀI THEO CHIỀU SÂU camera (VD 2 dãy trái/phải, mỗi dãy nhiều bàn
        # nối tiếp vào xa) cần bật True: khi đó "Dãy" được nhóm theo vị trí TRÁI/PHẢI (ax), còn "Ghế" đánh
        # số theo ĐỘ SÂU (gần/xa camera) trong cùng 1 dãy — NGƯỢC với mặc định (Dãy theo độ sâu, Ghế theo
        # trái/phải), vốn chỉ đúng với phòng xếp bàn thành nhiều hàng ngang cách đều camera.
        self.row_by_x = g("SEAT_ROW_BY_X", False)
        self.seat_far_first = g("SEAT_FAR_FIRST", False)  # chỉ dùng khi row_by_x=True: True = Ghe 1 la ban XA camera nhat
        self.reset()

    # ── học ────────────────────────────────────────────────────────────────
    def reset(self):
        self.calibrating = True
        self.seats = []
        self._t0 = None
        self._clusters = []
        self._n_updates = 0
        self.failed_rounds = 0

    def progress(self, now):
        if self._t0 is None:
            return 0.0
        return min(1.0, (now - self._t0) / max(self.calib_seconds, 1e-6))

    def update_calibration(self, now, dets, extras=None):
        """dets: list dict có "bbox". extras: list song song (dict yaw/pitch hoặc None).
        Trả về True đúng lúc học xong."""
        if not self.calibrating:
            return False
        if self._t0 is None:
            self._t0 = now
        self._n_updates += 1
        used = set()
        for i, d in enumerate(dets):
            ax, ay = anchor_of(d["bbox"], self.head_ratio)
            w = d["bbox"][2] - d["bbox"][0]
            h = d["bbox"][3] - d["bbox"][1]
            best, best_d = None, None
            for ci, c in enumerate(self._clusters):
                if ci in used:
                    continue
                dist = math.hypot(ax - c["ax"], ay - c["ay"])
                if dist <= self.cluster_radius_w * w and (best_d is None or dist < best_d):
                    best, best_d = ci, dist
            if best is None:
                self._clusters.append({"pts": [], "sizes": [], "extra": [], "row_ys": [], "ax": ax, "ay": ay})
                best = len(self._clusters) - 1
            c = self._clusters[best]
            used.add(best)
            c["pts"].append((ax, ay))
            c["sizes"].append((w, h))
            c["row_ys"].append(d["bbox"][3])  # đáy box (gần mặt bàn) — ổn định hơn đỉnh box khi camera đặt chéo
            if extras is not None and extras[i] is not None:
                c["extra"].append(extras[i])
            n = len(c["pts"])                       # tâm chạy để cụm bám theo vị trí thật
            c["ax"] += (ax - c["ax"]) / n
            c["ay"] += (ay - c["ay"]) / n
        if now - self._t0 >= self.calib_seconds and self._n_updates >= self.min_updates:
            return self.finalize(now)
        return False

    def finalize(self, now=None):
        need = max(2, self.min_hit_ratio * self._n_updates)
        kept = [c for c in self._clusters if len(c["pts"]) >= need]
        if not kept:                                  # chưa thấy ai ổn định -> học lại
            self.failed_rounds += 1
            self._clusters, self._n_updates = [], 0
            self._t0 = now
            return False
        seats = []
        for c in kept:
            ys = statistics.median(p[1] for p in c["pts"])
            xs = statistics.median(p[0] for p in c["pts"])
            ex = c["extra"]
            seats.append({
                "ax": xs, "ay": ys,
                "row_y": statistics.median(c["row_ys"]),
                "w": statistics.median(s[0] for s in c["sizes"]),
                "h": statistics.median(s[1] for s in c["sizes"]),
                "hits": len(c["pts"]),
                "yaw_base": statistics.median(e["yaw"] for e in ex) if ex else 0.0,
                "pitch_base": statistics.median(e["pitch"] for e in ex) if ex else 0.0,
            })
        self.seats = self._label_grid(_drop_merged(seats))
        self.calibrating = False
        return True

    def _group_by(self, seats, gap, key):
        seats = sorted(seats, key=lambda s: s[key])
        groups = [[seats[0]]]
        for s in seats[1:]:
            mean = sum(r[key] for r in groups[-1]) / len(groups[-1])
            if s[key] - mean > gap:
                groups.append([s])
            else:
                groups[-1].append(s)
        return groups

    def _infer_missing(self, rows, row_gap, col_gap, col_key="ax"):
        """Ghép các hàng thành LƯỚI chung theo cột (dựa trên trục `col_key` của TẤT CẢ chỗ đã thấy — mặc
        định "ax" cho bố cục nhiều hàng ngang; truyền "row_y" khi Dãy được nhóm theo trái/phải, Ghế theo
        độ sâu), rồi với mỗi hàng đang thiếu 1 cột mà đa số hàng khác có, chèn thêm 1 chỗ SUY LUẬN (nội
        suy vị trí từ hàng đó + cột đó), đánh dấu inferred=True. Chỉ hoạt động khi có ít nhất 2 hàng và
        mẫu lưới đủ rõ ràng (an toàn: không đủ rõ thì bỏ qua, không tự bịa thêm chỗ)."""
        if len(rows) < 2:
            return rows, 0
        all_seats = [s for row in rows for s in row]
        col_groups = self._group_by(all_seats, col_gap, key=col_key)
        col_groups.sort(key=lambda g: sum(s[col_key] for s in g) / len(g))
        col_centers = [sum(s[col_key] for s in g) / len(g) for g in col_groups]
        max_row_len = max(len(r) for r in rows)
        n_total = sum(len(r) for r in rows)
        # An toàn: chỉ suy luận khi (1) số cột đếm được KHỚP ĐÚNG với hàng đầy nhất — nếu dữ liệu là rải
        # rác ngẫu nhiên, số "cột" đếm được thường nhiều hơn hẳn số cột thật của hàng đầy nhất; và (2) lưới
        # đã đủ đặc (đa số ô đã có người), không suy luận từ dữ liệu quá thưa/mơ hồ.
        if max_row_len < 2 or len(col_centers) != max_row_len or n_total < 0.6 * len(rows) * max_row_len:
            return rows, 0  # lưới không đều/không rõ ràng -> không suy luận, tránh bịa sai
        other_key = "ay" if col_key == "ax" else "ax"
        med_w = statistics.median(s["w"] for s in all_seats)
        med_h = statistics.median(s["h"] for s in all_seats)
        added = 0
        new_rows = []
        for row in rows:
            have = set()
            for s in row:
                nearest = min(range(len(col_centers)), key=lambda i: abs(col_centers[i] - s[col_key]))
                if abs(col_centers[nearest] - s[col_key]) <= col_gap:
                    have.add(nearest)
            row_other = sum(s[other_key] for s in row) / len(row)
            new_row = list(row)
            for i, cx in enumerate(col_centers):
                if i not in have:
                    new_seat = {"ax": 0.0, "ay": 0.0, "w": med_w, "h": med_h, "hits": 0,
                               "yaw_base": 0.0, "pitch_base": 0.0, "inferred": True}
                    new_seat[col_key], new_seat[other_key] = cx, row_other
                    if col_key == "row_y":
                        # Dãy theo trái/phải: cột = độ sâu (row_y, đáy box) -> suy ra điểm neo ay từ khoảng cách
                        # row_y - ay của các chỗ thật cùng cột (trước đây ay giữ 0 -> khung "ma" ở mép trên ảnh)
                        new_seat["ay"] = cx - statistics.median(s["row_y"] - s["ay"] for s in col_groups[i])
                    new_seat.setdefault("row_y", new_seat["ay"])
                    new_row.append(new_seat)
                    added += 1
            new_rows.append(new_row)
        return new_rows, added

    def _label_grid(self, seats):
        for s in seats:
            s.setdefault("row_y", s["ay"])   # tương thích ngược: chỗ nạp từ bản đồ cũ/chỉnh tay chưa có row_y
        row_gap = self.row_gap_h * statistics.median(s["h"] for s in seats)
        col_gap = self.cluster_radius_w * statistics.median(s["w"] for s in seats)

        if self.row_by_x:
            # Bố cục DÃY KÉO DÀI THEO CHIỀU SÂU (VD 2 dãy trái/phải, mỗi dãy nhiều bàn nối tiếp vào xa):
            # "Dãy" nhóm theo trái/phải (ax), "Ghế" đánh số theo độ sâu (row_y) trong cùng 1 dãy.
            primary_key, secondary_key, gap_primary, gap_secondary = "ax", "row_y", col_gap, row_gap
        else:
            # Mặc định: bố cục nhiều hàng ngang cách đều camera — "Dãy" nhóm theo độ sâu (row_y, đáy box,
            # ổn định hơn đỉnh box khi camera đặt chéo), "Ghế" đánh số trái/phải (ax).
            primary_key, secondary_key, gap_primary, gap_secondary = "row_y", "ax", row_gap, col_gap

        rows = self._group_by(seats, gap_primary, key=primary_key)
        rows.sort(key=lambda r: sum(s[primary_key] for s in r) / len(r),
                  reverse=(self.row_by_x and self.col_right_to_left))   # chiều đánh số Dãy khi Dãy = trái/phải
        if getattr(self, "grid_complete", True):
            rows, n_added = self._infer_missing(rows, gap_primary, gap_secondary, col_key=secondary_key)
            if n_added:
                print(f"[SeatMap] Suy luận thêm {n_added} chỗ trống lúc học (không có ai ngồi lúc đó) theo quy luật lưới.")
        out = []
        for ri, row in enumerate(rows, start=1):
            # Ghế theo độ sâu: y lớn = gần camera -> mặc định (seat_far_first=False) sắp y GIẢM dần để Ghế 1 là bàn
            # gần camera nhất (trước đây sắp tăng dần -> Ghế 1 thành bàn xa nhất, ngược với SEAT_FAR_FIRST)
            secondary_reverse = (not self.seat_far_first) if self.row_by_x else self.col_right_to_left
            ordered = sorted(row, key=lambda s: s[secondary_key], reverse=secondary_reverse)
            for ci, s in enumerate(ordered, start=1):
                s.setdefault("inferred", False)
                s["row"], s["col"] = ri, ci
                s["label"] = f"D{ri}-G{ci}"
                out.append(s)
        for i, s in enumerate(out):
            s["idx"] = i
        return out

    # ── dùng sau khi khoá ─────────────────────────────────────────────────
    def assign(self, dets, offset=(0.0, 0.0)):
        """Trả về list seat_idx (hoặc None) song song với dets. 1 chỗ nhận tối đa 1 người.
        offset: độ dịch (px) của toàn bộ sơ đồ do camera bị xê dịch (xem SeatPipeline._update_cam_offset)."""
        ox, oy = offset
        res = [None] * len(dets)
        if self.calibrating or not self.seats or not dets:
            self._last_anchor = {}
            return res
        anchors = [anchor_of(d["bbox"], self.head_ratio) for d in dets]
        cost = np.zeros((len(dets), len(self.seats)))
        for i, (ax, ay) in enumerate(anchors):
            for j, s in enumerate(self.seats):
                dist = math.hypot(ax - (s["ax"] + ox), ay - (s["ay"] + oy)) / max(s["w"], 1.0)
                last = self._last_anchor.get(j)
                if last is not None and math.hypot(ax - last[0], ay - last[1]) <= self.match_radius_w * max(s["w"], 1.0):
                    dist *= self.sticky_discount   # vẫn gần vị trí lần trước -> ưu tiên giữ nguyên ghép cũ
                cost[i, j] = dist
        rows, cols = linear_sum_assignment(cost)
        new_last = {}
        for i, j in zip(rows, cols):
            if cost[i, j] <= self.match_radius_w:
                res[i] = int(j)
                new_last[j] = anchors[i]
        self._last_anchor = new_last   # chỗ không được ghép tick này thì quên trí nhớ cũ (hợp lý: đã rời hẳn)
        return res

    def label(self, idx):
        if idx is None:
            return "Dang hoc vi tri..." if self.calibrating else "Cho moi (chua hoc)"
        return self.seats[idx]["label"]

    def to_dict(self):
        return {"seats": [{k: v for k, v in s.items()} for s in self.seats]}

    @classmethod
    def from_dict(cls, d, cfg=None):
        m = cls(cfg)
        m.seats = d["seats"]
        m.calibrating = False
        return m

    @classmethod
    def from_dict_scaled(cls, d, frame_w, frame_h, cfg=None, tol=2):
        """Như from_dict nhưng đối chiếu với kích thước khung hình THẬT: bản đồ gán tay tính theo pixel của
        khung hình lúc gán (lưu trong d["frame_size"]=[W,H]); nếu khung hình dùng để trích dữ liệu/chạy thật
        có độ phân giải khác thì quy đổi toạ độ cho khớp, thay vì để chỗ ngồi lệch âm thầm.
        Trả về (SeatMap, ghi_chu): ghi_chu là None nếu không cần quy đổi, hoặc chuỗi mô tả."""
        size = d.get("frame_size")
        if not size or not frame_w or not frame_h:
            note = None if size else "bản đồ không ghi kích thước khung hình gốc nên không kiểm tra được độ lệch"
            return cls.from_dict(d, cfg), note
        mw, mh = float(size[0]), float(size[1])
        if abs(mw - frame_w) <= tol and abs(mh - frame_h) <= tol:
            return cls.from_dict(d, cfg), None
        sx, sy = frame_w / mw, frame_h / mh
        scaled = dict(d, seats=rescale_seats(d["seats"], sx, sy))
        note = (f"bản đồ gán trên khung {int(mw)}x{int(mh)} nhưng khung hình hiện tại là "
                f"{int(frame_w)}x{int(frame_h)} -> đã quy đổi toạ độ (x{sx:.3f}, y{sy:.3f})")
        return cls.from_dict(scaled, cfg), note


def rescale_seats(seats, sx, sy):
    """Quy đổi toạ độ/kích thước các chỗ ngồi khi đổi độ phân giải khung hình (không sửa bản gốc)."""
    out = []
    for s in seats:
        t = dict(s)
        t["ax"] = s["ax"] * sx
        t["ay"] = s["ay"] * sy
        t["w"] = s["w"] * sx
        t["h"] = s["h"] * sy
        if "row_y" in s:
            t["row_y"] = s["row_y"] * sy
        out.append(t)
    return out


# ════════════════════════════════════════════════════════════════════════════════════════════
# CHỈNH TAY BẢN ĐỒ CHỖ NGỒI — logic THUẦN (không giao diện, kiểm thử được)
# Dùng bởi seat_map_editor.py (công cụ chỉnh bằng chuột trên video) và phím E trong giamsatthi.py.
# (Trước đây nằm ở file riêng seat_editor_core.py, đã gộp vào đây cho gọn.)
# ════════════════════════════════════════════════════════════════════════════════════════════

def nearest_seat(seats, x, y, max_dist=None):
    """Chỗ gần điểm (x,y) nhất. max_dist: chỉ tính nếu trong bán kính đó (đơn vị: bề rộng seat), else None."""
    if not seats:
        return None
    best, best_d = None, None
    for i, s in enumerate(seats):
        d = math.hypot(x - s["ax"], y - s["ay"])
        if max_dist is not None and d > max_dist * max(s["w"], 1.0):
            continue
        if best_d is None or d < best_d:
            best, best_d = i, d
    return best


def add_seat(seats, x, y, default_w=90.0, default_h=160.0):
    """Thêm 1 chỗ mới tại (x,y). Kích thước lấy theo trung vị các chỗ đã có, nếu chưa có chỗ nào thì dùng mặc định."""
    if seats:
        ws = sorted(s["w"] for s in seats)
        hs = sorted(s["h"] for s in seats)
        w, h = ws[len(ws) // 2], hs[len(hs) // 2]
    else:
        w, h = default_w, default_h
    seats.append({"ax": float(x), "ay": float(y), "w": float(w), "h": float(h), "hits": 0,
                 "yaw_base": 0.0, "pitch_base": 0.0, "inferred": False, "manual": True})
    return len(seats) - 1


def move_seat(seats, idx, x, y):
    seats[idx]["ax"], seats[idx]["ay"] = float(x), float(y)


def remove_seat(seats, idx):
    seats.pop(idx)


def relabel(seats, row_gap_h=0.6, row_by_x=False, col_right_to_left=False, seat_far_first=False):
    """Gán lại nhãn Dãy-Ghế theo vị trí hiện tại (gọi sau khi thêm/xoá/di chuyển).
    row_by_x=False (mặc định): Dãy nhóm theo chiều sâu (ay), Ghế đánh số trái/phải — đúng cho phòng xếp
    nhiều hàng ngang cách đều camera.
    row_by_x=True: Dãy nhóm theo trái/phải (ax), Ghế đánh số theo chiều sâu — đúng cho phòng xếp bàn
    thành vài dãy kéo dài vào chiều sâu camera (VD 2 dãy trái/phải, mỗi dãy nhiều bàn nối tiếp vào xa)."""
    if not seats:
        return seats
    primary_key = "ax" if row_by_x else "ay"
    secondary_key = "ay" if row_by_x else "ax"
    gap = row_gap_h * (sorted(s["h"] for s in seats)[len(seats) // 2])
    ordered = sorted(seats, key=lambda s: s[primary_key])
    rows = [[ordered[0]]]
    for s in ordered[1:]:
        mean = sum(r[primary_key] for r in rows[-1]) / len(rows[-1])
        rows.append([s]) if s[primary_key] - mean > gap else rows[-1].append(s)
    if row_by_x and col_right_to_left:
        rows.sort(key=lambda r: sum(s[primary_key] for s in r) / len(r), reverse=True)
    out = []
    for ri, row in enumerate(rows, start=1):
        reverse = (not seat_far_first) if row_by_x else col_right_to_left   # như _label_grid: y lớn = gần camera
        for ci, s in enumerate(sorted(row, key=lambda s: s[secondary_key], reverse=reverse), start=1):
            s["row"], s["col"] = ri, ci
            s["label"] = f"D{ri}-G{ci}"
            out.append(s)
    for i, s in enumerate(out):
        s["idx"] = i
    seats[:] = out
    return seats