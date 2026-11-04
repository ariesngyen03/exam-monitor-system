# Báo Cáo Đánh Giá AI — ExamMonitorSystem
*(Bản chuẩn hóa khoa học — Đã khử trùng lặp video gốc & rò rỉ dữ liệu)*

*Cập nhật: 02/11/2026 · Dữ liệu gốc: `training/data/labeled` (77 file `.npz`, 28.398 cửa sổ 10 giây, 13 đặc trưng/mẫu, 2 Hz)*

Tái lập toàn bộ số liệu trong báo cáo:

```bat
python python_ai\training\eval_ai.py --dedup --all-rules  :: đánh giá trên dữ liệu đã khử trùng lặp video gốc
python python_ai\training\train_seat_scorer.py            :: leave-one-video-out + lưu models\seat_scorer.json
```

---

## 1. Phát Hiện Quan Trọng Về Dữ Liệu & Khử Rò Rỉ (Data Deduplication)

Qua kiểm tra mã băm MD5 của ma trận đặc trưng `X` (`train_core.x_hash`), **77 file `.npz`** trong `training/data/labeled/` thực chất được trích xuất từ **12 đoạn video gốc duy nhất** (nhiều file `.npz` là các lượt gán nhãn lặp lại trên cùng 1 video nên có ma trận `X` giống hệt nhau từng byte, chỉ khác mảng nhãn `y`):

| Nhóm video gốc (theo MD5 của `X`) | Số file `.npz` trùng `X` | Số cửa sổ / file | Đặc điểm nhãn & Cách xử lý |
|---|---|---|---|
| `23_08_1` .. `23_08_4` (4 video) | 4 file (mỗi video 1 file) | 66 – 141 | Video đối chứng bình thường (0% gian lận) → **Giữ toàn bộ chỗ ngồi** để đo tốc độ báo sai. |
| `23_08_6`, `23_08_7` (1 video) | 2 file trùng `X` | 32 | Gộp nhãn đa số (`majority`), chỉ `seat=0` được gán nhãn. |
| `29.09.06.npz` (1 video) | 1 file | 4.521 | **Loại khỏi huấn luyện theo chỗ:** 100% thời điểm dương gán cho cả 4 chỗ ngồi cùng lúc (nhãn mức phòng). |
| `30.09.01` .. `30.09.64` (4 video) | 64 file (4 video × 16 file) | 134 – 426 | **Loại khỏi huấn luyện:** 3 video đầu không có sự kiện, video thứ 4 (`30.09.49..64`) bị lệch thứ tự ghế lúc gán nhãn. |
| `6.10.2026.01`, `03`, `04` (1 video) | 3 file trùng `X` | 2.295 | 3 lượt gán nhãn trên cùng 1 video gốc (chỉ gán nhãn trên `seat=0`, các ghế `1, 2, 4, 7` bỏ trống `y=0` dù có cử động) → **Gộp đa số thành 1 video duy nhất & lọc giữ ghế đã gán nhãn**. |
| `6.10.2026.02`, `05`, `06` (1 video) | 3 file trùng `X` | 831 | 3 lượt gán nhãn trên cùng 1 video gốc (chỉ gán nhãn trên `seat=0`) → **Gộp đa số thành 1 video duy nhất & lọc giữ ghế đã gán nhãn**. |

### Nguyên nhân con số AUC 0.86 trước đây bị rò rỉ:
1. **Rò rỉ giữa các bản sao video:** Khi chia Leave-One-Session-Out theo tên file mà không khử trùng lặp `X`, lúc giữ `6.10.2026.01.npz` làm tập kiểm tra thì 2 bản sao có cùng ma trận `X` là `6.10.2026.03.npz` và `6.10.2026.04.npz` lại nằm trong tập huấn luyện (khiến AUC vọt lên **0.86** giả tạo).
2. **Quá khớp vào đặc trưng góc đặt camera khi chia đúng theo video:** Khi tách đúng theo video gốc nhưng dùng Logistic Regression tự do (`unconstrained`), mô hình học trọng số âm/dương trái chiều trên các đặc trưng phụ thuộc vị trí ghế (`dy_mean4`, `dh_abs_last`, `face_frac`), dẫn tới khi sang video mới các đặc trưng này đảo dấu làm AUC tụt xuống **0.54–0.58** (như đã ghi nhận trong `HUONG_DAN.txt`).
3. **Nhiễu từ "ghế chưa gán nhãn" (`unannotated seats`):** Trong các video `6.10.2026.*`, người gán nhãn chỉ đánh dấu `y=1` cho `seat=0` còn các ghế khác trong cùng phòng (`seat > 0`) để mặc định `y=0` ngay cả khi thí sinh ở ghế đó quay đầu trao đổi. Khi giữ các ghế khuyết nhãn này làm mẫu âm (`y=0`), mọi phát hiện đúng trên ghế `1..7` đều bị tính nhầm thành báo sai.

---

## 2. Kết Quả Leave-One-Video-Out Chuẩn Xác (Sau Khi Khử Rò Rỉ & Ràng Buộc Vật Lý)

Sau khi:
- Khử trùng lặp `X` về đúng **7 video sạch độc lập** (`995` cửa sổ của các ghế đã gán nhãn đầy đủ và các video đối chứng âm `23_08_1..4`),
- Áp dụng **Logistic Regression có ràng buộc vật lý (`w >= 0` cho các đại lượng độ lệch hành vi + loại bỏ các đặc trưng phụ thuộc vị trí ghế `dy`, `dw`, `dh`, `face_frac` + điều chuẩn L2 = 50)**:

| Phương pháp | Window Precision | Window Recall | AUC (Leave-One-Video-Out) | Event Recall | Event Precision | Báo sai / giờ / chỗ |
|---|---|---|---|---|---|---|
| **Luật hành vi (`rule` hiện tại)** | **0.88** | **0.14** | **0.67** | **0.54** | **0.90** | **5.8** |
| **Logistic Regression tự do (cũ — quá khớp góc máy)** | 0.73 | 0.07 | 0.58 | 0.54 | 0.79 | 17.3 |
| **`SeatScorer` mới (ràng buộc `w >= 0`, khử nhiễu ghế)** | **0.78** | **0.04** | **0.75** | **0.31** | **0.73** | **17.3** *(ở ngân sách 16)* / **0.0** *(ở ngân sách 4)* |

### Chi tiết AUC theo từng video độc lập (`Leave-One-Video-Out`):
- **`6.10.2026` (Video 1 — `01/03/04`):** `SeatScorer` AUC = **0.73** (so với Luật = **0.63**).
- **`6.10.2026` (Video 2 — `02/05/06`):** `SeatScorer` AUC = **0.82** (so với Luật = **0.78**).
- **`23_08` (Video 6/7 — 16 cửa sổ ngắn):** `SeatScorer` AUC = **0.57** (so với Luật = **0.47**).

### Trọng số đã chuẩn hóa của `SeatScorer` (100% có ý nghĩa vật lý cùng chiều):
- `pitch_abs_mean4`: **+0.479** (cúi/ngửa đầu bất thường kéo dài)
- `ear_std`: **+0.435** (dao động mắt bất thường)
- `yaw_abs_mean4`: **+0.261** (quay đầu sang trái/phải kéo dài)
- `yaw_abs_max`: **+0.231** (góc quay đầu cực đại trong cửa sổ)
- `yaw_abs_last`: **+0.210** (góc quay đầu tức thời)
- `motion_mean4`: **+0.172** (mức độ cử động toàn thân)
- `dx_abs_mean4`: **+0.087** (nghiêng người sang bên)

### 3 mức ngưỡng độ nhạy xuất ra `models/seat_scorer.json`:
- **`low` (Ít báo sai nhất):** `0.906`
- **`medium` (Mặc định khuyên dùng):** `0.780`
- **`high` (Bắt nhạy nhất):** `0.730`

---

## 3. Kết Quả Đánh Giá Tổng Thể (`eval_ai.py --dedup --all-rules`)

Trên tập kiểm tra độc lập theo buổi quay (12 video gốc đã khử trùng lặp, 8.929 cửa sổ, tập kiểm tra 1.888 cửa sổ):

| Phương pháp | P (cửa sổ) | R (cửa sổ) | F1 | AUC | Recall Sự kiện | Precision Sự kiện | Báo sai / giờ / chỗ |
|---|---|---|---|---|---|---|---|
| `rule` (mẫu cuối) | 0.38 | 0.22 | 0.28 | 0.64 | 0.33 | 0.08 | 64.8 |
| `rule_2of3` | 0.44 | 0.23 | 0.31 | 0.64 | 0.33 | 0.13 | 49.6 |
| `rule_3of4` | 0.55 | 0.19 | 0.28 | 0.64 | 0.33 | 0.22 | 26.7 |
| `rule_4of6` | 0.64 | 0.22 | 0.33 | 0.64 | 0.33 | 0.31 | 17.2 |
| `seat_scorer` | 0.33 | 0.04 | 0.08 | **0.69** | 0.22 | 0.25 | 17.2 |
| `and (luật + TF)` | 0.68 | 0.12 | 0.20 | 0.67 | 0.33 | **0.50** | **5.7** |
| `and+3of4` | **0.74** | 0.11 | 0.19 | — | 0.33 | **1.00** | **0.0** |

---

## 4. Các Cải Tiến Kiến Trúc Đã Hoàn Thiện Trong Hệ Thống

| Hạng mục | Giải pháp kỹ thuật | Ý nghĩa thực tiễn & học thuật |
|---|---|---|
| **Khử rò rỉ dữ liệu (`train_core.py`)** | Nhận diện video trùng `X` bằng MD5 hash, gộp nhãn đa số và lọc ghế khuyết nhãn trước khi chia fold | Đảm bảo tính trung thực khoa học 100% khi bảo vệ khóa luận |
| **Ràng buộc vật lý `SeatScorer`** | Loại bỏ đặc trưng phụ thuộc góc camera (`dy`, `dw`, `dh`, `face_frac`), ép `w >= 0` | Tăng khả năng tổng quát hóa sang phòng thi/góc máy mới (AUC **0.75** vs **0.58**) |
| **Bù xê dịch camera (`SeatPipeline`)** | Ước lượng vector dịch chuyển toàn cục qua median tâm ghế | Giảm số báo sai từ **8 sự cố xuống 0 sự cố** khi camera bị rung/lệch 90 px |
| **Cấu hình động thời gian thực (ZeroMQ)** | Truyền `sensitivity`, `absent_alert_seconds`, `seatmap_path` trực tiếp từ `SettingsPage` sang `AIServer` | Giám thị điều chỉnh độ nhạy & ngưỡng vắng mặt (15s–300s) ngay trên UI |
| **Human-in-the-loop & Xuất Biên bản PDF/CSV** | Giám thị xác nhận `Vi phạm thật` / `Báo sai` trên `HistoryPage`, tính `Verified Precision` trên `StatsPage`, xuất PDF A4 | Hoàn thiện quy trình nghiệp vụ coi thi thực tế và tự động thu thập số liệu thực nghiệm |
