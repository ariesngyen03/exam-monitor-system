# python_ai — AI Server

Nhận khung hình từ C++ Client qua ZeroMQ, phân tích hành vi theo **chỗ ngồi**, trả kết quả về giao diện.
Hệ thống là **công cụ gợi ý chỗ cần chú ý** cho giám thị, không phải công cụ kết luận gian lận. Xem
[training/BAO_CAO_DANH_GIA_AI.md](training/BAO_CAO_DANH_GIA_AI.md) để biết lý do và số liệu.

```
C++ VideoCapture ─▶ ZmqPublisher  ──FramePacket (JPEG, phòng, giám thị, reset_seq) :5555──▶ main.py
                     (giới hạn ~4 frame/giây)                                                │
                                                     ExamAIService (2 nhịp/giây)            │
                                                       YOLO người → sơ đồ chỗ ngồi (+ bù xê dịch camera)
                                                       → Face Mesh → đặc trưng 13 chiều / chỗ
                                                       → SeatScorer quyết định "cần chú ý"; luật giải thích
                                                     ExamSession: log CSV, báo cáo JSON, ảnh bằng chứng
C++ UI ◀── ZmqSubscriber ◀──InferenceResult :5556 (+ heartbeat mỗi 1 s) ────────────────────┘
```

## Yêu cầu

- **Python 3.10 / 3.11** (`mediapipe.solutions` không có trên 3.12+).
- `pip install -r python_ai/requirements.txt` (hoặc `scripts\setup_venv.bat`).
- Dò người chạy bằng OpenVINO (`models/yolov8s_openvino_544x960/`, cùng trọng số yolov8s): GPU tích hợp Intel nếu có
  (~30–45 ms/nhịp, không chiếm CPU), không thì CPU. Thiếu `openvino` thì tự quay về PyTorch (~200 ms/nhịp).
  Xuất lại sau khi đổi trọng số: `python python_ai\tools\export_openvino.py`.
- Camera giật: `python python_ai\tools\camera_check.py --index 0` đo định dạng (MJPG/YUY2) và fps thật.

## Chạy

```bat
scripts\run.bat                                       :: AI server + giao diện C++
python python_ai\main.py --help                       :: toàn bộ tham số
python python_ai\main.py --sensitivity low            :: ít báo sai nhất (low | medium | high)
```

- Phòng thi, giám thị, camera hoặc file video, và nút **Học lại sơ đồ chỗ ngồi** nằm ở trang **Cài đặt** của giao diện.
- Bấm **BẮT ĐẦU**: 15 giây đầu AI học sơ đồ chỗ ngồi. Bấm **DỪNG LẠI**: sau 5 giây AI đóng phiên và ghi báo cáo, trang Lịch sử và Thống kê tự cập nhật.
- Dữ liệu ghi vào `data/` (hoặc `EXAM_DATA_DIR`): `evidence/<ngày>/<phòng>/`, `reports/<ngày>.json`, `logs/`.
- Nhận diện cảm xúc **tắt mặc định** (bật bằng `--emotion`, cần `pip install deepface tf-keras`).

Kiểm thử không cần build C++:

```bat
python python_ai\main.py
python python_ai\tools\mock_cpp_client.py --source video_hoac_anh.jpg --seconds 40
```

## Đánh giá & huấn luyện

```bat
python python_ai\training\eval_ai.py --all-rules      :: so sánh luật / Transformer / hợp nhất, theo sự kiện
python python_ai\training\train_seat_scorer.py        :: leave-one-session-out + lưu models\seat_scorer.json
```

Dữ liệu đặc trưng đã gán nhãn nằm ở `training/data/labeled/`, chép từ repo AI (`train_data/labeled`).

## Cấu trúc

| File | Vai trò |
|---|---|
| `main.py` | ZeroMQ server, vòng đời phiên, đóng gói Protobuf |
| `session.py` | Log, sự cố, ảnh bằng chứng của 1 phiên |
| `exam_ai_service.py` | Dịch vụ suy luận; ánh xạ vi phạm → `behavior_type` |
| `seat_scorer.py` | Mô hình chấm điểm chỗ ngồi (dùng chung cho huấn luyện và chạy thật) |
| `behavior.py` | BehaviorEngine: SeatScorer quyết định, luật giải thích, chống nhiễu |
| `pipeline.py` | 1 nhịp phân tích + **bù xê dịch camera** |
| `ai_core.py`, `seat_map.py`, `features.py`, `face_analyzer.py`, `person_detector.py`, `emotion_worker.py`, `tsn_worker.py`, `giamsatthi.py`, `seat_map_editor.py` | Lõi AI, gốc từ `giamsat_v11_code` |
| `training/` | `eval_ai.py`, `train_seat_scorer.py`, `train_core.py`, `train_transformer.py`, dữ liệu, báo cáo |
| `models/` | `seat_scorer.json`, `transformer_seat.pt` + meta, `yolov8n.pt` |

**Đã sửa so với bản gốc** (khi chép lõi AI mới từ repo AI về, không chép đè các file sau):
`exam_ai_service.py`, `behavior.py` (thêm SeatScorer), `pipeline.py` và `seat_map.py` (bù xê dịch camera).

## behavior_type gửi sang C++

| Giá trị | Nguồn |
|---|---|
| `Normal` / `Empty` | Không cần chú ý (có người / chỗ trống) |
| `Turning Head` | Quay đầu, nhìn lệch |
| `Head Down` | Cúi/ngửa đầu bất thường |
| `Leaning` | Nghiêng người / sang chỗ khác |
| `Absent` | Rời chỗ quá `SEAT_ABSENT_ALERT_SECONDS` |
| `Suspicious` | SeatScorer báo nhưng luật không chỉ ra tư thế cụ thể |
| `Calibrating` / `Unassigned` | Đang học chỗ / người chưa khớp chỗ nào |

- `confidence` là **điểm chú ý** (0..1) dùng để xếp hạng chỗ ngồi, **không phải xác suất gian lận**.
- `new_alert` bật khi có ảnh bằng chứng mới; `new_incident` phân biệt sự cố mới với ảnh bổ sung cho sự cố đang diễn ra.
