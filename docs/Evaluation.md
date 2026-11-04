# Báo Cáo Đánh Giá Toàn Diện & Kế Hoạch Hoàn Thiện Khóa Luận (14 Ngày)
**Đề tài:** Hệ thống Giám sát Phòng thi Thông minh Hỗ trợ Giám thị (ExamMonitorSystem)  
**Cập nhật:** 04/11/2026

---

## Phần 1. Đánh Giá Thực Trạng Hệ Thống (Kiểm Toán Mã Nguồn Thực Tế)

### 1.1. Kiến trúc kỹ thuật thực tế (Đính chính chuẩn xác theo mã nguồn)
Hệ thống được thiết kế theo mô hình **Edge Client–Server nội bộ (Decoupled Multi-Process Architecture)**, tách biệt hoàn toàn giữa giao diện hiển thị tốc độ cao và tiến trình suy luận AI nặng:

- **Desktop Client (`cpp_client/` — C++20 & Qt6 Widgets + OpenCV 4.12):**
  - Đọc luồng hình ảnh từ Webcam USB / Video / RTSP ở định dạng **MJPG 1080p @ 30 FPS** trên luồng riêng (`VideoThread`), giảm tải CPU nhờ hạ độ phân giải gói tin trước khi nén JPEG gửi cho AI.
  - Giao diện 5 phân trang chuyên biệt: **Tổng quan (`DashboardPage`)**, **Video trực tiếp (`LiveViewPage`)**, **Lịch sử sự cố (`HistoryPage`)**, **Thống kê (`StatsPage`)**, và **Cài đặt (`SettingsPage`)**.
  - Lưu trữ và quản lý báo cáo qua `ReportStore` (`data/reports/YYYY-MM-DD.json`), hỗ trợ quy trình **Human-in-the-loop** (Giám thị xác nhận *Vi phạm thật* / *Báo sai*, ghi chú SBD, xuất **Biên bản PDF chuẩn A4** bằng `QPdfWriter` và **CSV chuẩn UTF-8 BOM**).
- **Cơ chế Giao tiếp Liên tiến trình (`proto/exam_monitor.proto` — ZeroMQ + Protocol Buffers):**
  - Kênh **PUB/SUB cổng `5555` (`FramePacket`)**: C++ Client gửi ảnh JPEG kèm cấu hình thời gian thực (`room_id`, `proctor_id`, `reset_seq`, `sensitivity`, `absent_alert_seconds`, `seatmap_path`) sang AI Server (giới hạn ~4 frame/giây để chống nghẽn).
  - Kênh **PUB/SUB cổng `5556` (`InferenceResult`)**: Python AI Server trả về danh sách tọa độ người/ghế, hành vi (`behavior_type`), điểm chú ý (`confidence`), cờ sự cố mới (`new_alert`, `new_incident`), đường dẫn ảnh bằng chứng và gói tin `Heartbeat` mỗi 1 giây.
- **Headless AI Server (`python_ai/` — Python 3.11 + OpenVINO + MediaPipe + PyTorch):**
  - **Tầng 1 — Phát hiện người (`person_detector.py`)**: Mô hình **YOLOv8s** biên dịch sang **OpenVINO FP16 (`544×960`)**, tự động ưu tiên chạy trên **Intel iGPU** (~30–45 ms/nhịp) hoặc fallback về CPU (~190–220 ms/nhịp).
  - **Tầng 2 — Ánh xạ sơ đồ ghế & Bù lệch camera (`seat_map.py`, `pipeline.py`)**: Tự động học vị trí ghế trong 15 giây đầu (hoặc nạp `seatmap.json` cố định), ghép cặp người–ghế bằng thuật toán **Hungarian Matching** kết hợp **bù vector xê dịch camera toàn cục** (qua median độ lệch tâm các ghế có người).
  - **Tầng 3 — Phân tích khuôn mặt & Tư thế đầu 3D (`face_analyzer.py`, `features.py`)**: Sử dụng **MediaPipe FaceMesh (478 landmarks)** giải bài toán **Perspective-n-Point (`cv2.solvePnP`)** ước lượng góc quay đầu 3D (`Yaw`, `Pitch`, `Roll`), tỉ lệ mở mắt (`EAR`), độ mở miệng (`MAR`) và trích xuất vector **13 đặc trưng hành vi** ở tần số **2 Hz**.
  - **Tầng 4 — Đánh giá hành vi & Chấm điểm chỗ ngồi (`behavior.py`, `seat_scorer.py`)**: Kết hợp mô hình **SeatScorer** (Logistic Regression có ràng buộc đơn điệu vật lý $w \ge 0$ trên cửa sổ trượt 20 mẫu = 10 giây) để xếp hạng mức độ cần chú ý và **Bộ luật giải thích (`rule_evaluate` + bộ lọc chống nhiễu $k\text{-of-}n$)** để gán nhãn hành vi cụ thể (`Turning Head`, `Head Down`, `Leaning`, `Absent`, `Suspicious`).

---

### 1.2. Trạng thái ổn định của UI & Lõi AI (Sau 5 Phase nâng cấp ngày 02/11/2026)

| Thành phần | Trạng thái hiện tại | Kết quả kiểm chứng |
|---|---|---|
| **Lõi AI & Mô hình (`python_ai/`)** | **Ổn định (4/4 Unit Tests PASS)** | Đã khử trùng lặp 77 file `.npz` về 12 video gốc độc lập; `SeatScorer` đạt **AUC = 0.75 ± 0.20** (vượt bộ luật cố định **0.67**); bù lệch camera giảm báo sai từ **8 xuống 0 sự cố**. |
| **Giao diện C++ Qt6 (`build/ExamMonitorSystem.exe`)** | **Đã biên dịch bản Release mới nhất (02/11/2026)** | Đầy đủ cấu hình động (Độ nhạy AI, Thời gian rời ghế, Mở `seat_map_editor.py`), Xác minh cảnh báo Đúng/Sai, Thống kê `Verified Precision`, Xuất PDF/CSV. |
| **Bản đóng gói (`dist/ExamMonitorSystem/`)** | **CẦN CẬP NHẬT (Đang ở bản cũ 20/10/2026)** | File `.exe` và thư mục `python_ai` trong `dist/` chưa được đóng gói lại sau khi nâng cấp ngày 02/11/2026. Cần chạy lại `scripts\package.bat`. |
| **Dữ liệu mẫu thực nghiệm (`data/reports/`)** | **ĐANG TRỐNG** | Cần chạy 1–2 phiên kịch bản mẫu (10 phút) với video hoặc webcam để sinh file JSON, ảnh bằng chứng, biên bản PDF và CSV phục vụ chụp ảnh minh họa cho Chương 4. |
| **Thư mục gốc (`models/`, `yolov8n.pt`)** | **Dư thừa file rác cũ** | Hệ thống thật sử dụng `python_ai/models/`. Thư mục `models/` ở gốc (`transformer_seat.pt.tat`) là bản lưu cũ từ giai đoạn thử nghiệm ban đầu, nên dọn dẹp để tránh nhầm lẫn. |

---

## Phần 2. Quyết Định Chiến Lược Về Phạm Vi Sản Phẩm (Trong 2 Tuần Cuối)

### Câu hỏi: Có nên làm thêm App Điện thoại, Web/Server không? Đóng gói sản phẩm như thế nào?
1. **KHÔNG làm thêm App Điện thoại hay Web Server trong 2 tuần cuối:**
   - **Lý do 1 (Bảo toàn thời gian cho Chương 2, 3, 4):** Trong 14 ngày cuối, 70% điểm số của khóa luận nằm ở chất lượng cuốn báo cáo (cơ sở lý thuyết, công thức toán học, sơ đồ kiến trúc, bảng biểu thực nghiệm) và sự trơn tru khi demo trực tiếp.
   - **Lý do 2 (Tránh mở thêm điểm yếu khi phản biện):** Một Web/Mobile App viết vội trong vài ngày sẽ thiếu bảo mật, thiếu xử lý mất kết nối và lệch kiến trúc Edge AI hiện tại, dễ bị Hội đồng xoáy sâu vào các lỗi phụ thay vì ghi nhận chiều sâu của lõi Computer Vision + C++ Qt6.
   - **Lý do 3 (Cách xử lý khéo léo trong báo cáo):** Đưa mô hình *"Web Server quản trị tập trung đa phòng thi & Ứng dụng di động cho giám thị hành lang"* vào **Chương 5 — Hướng phát triển của đề tài** kèm 1 sơ đồ kiến trúc mở rộng. Hội đồng sẽ đánh giá cao tư duy hệ thống của bạn mà không đòi hỏi bạn phải demo phần đó.
2. **Đóng gói như một Sản phẩm Desktop Chuyên nghiệp (`Portable Product Package`):**
   - Sử dụng `scripts/package.py` để tạo gói sản phẩm độc lập hoàn chỉnh tại `dist/ExamMonitorSystem/`:
     - Tích hợp sẵn `app/ExamMonitorSystem.exe` + toàn bộ DLL Qt6, OpenCV, ZeroMQ.
     - Tích hợp sẵn `python/` (Python 3.11 Portable + OpenVINO + MediaPipe + PyTorch đã cắt gọn các gói thừa).
     - Tích hợp cơ chế **Tự động giám sát và phục hồi tiến trình (`Start.bat` + `ai_watchdog.bat`)**: tự khởi động ngầm AI Server, tự khởi động lại nếu AI bị tắt đột ngột, và tự động lưu báo cáo + đóng AI Server khi giám thị tắt giao diện Qt.

---

## Phần 3. Đề Cương Chi Tiết Báo Cáo Khóa Luận (Từ Chương 2 Đến Chương 5)

Để viết nhanh và đạt điểm tối đa, bạn hãy bám sát cấu trúc chuẩn dưới đây (mọi công thức, sơ đồ và bảng số liệu đều đã có sẵn trong code của dự án):

### CHƯƠNG 2: CƠ SỞ LÝ THUYẾT VÀ CÔNG NGHỆ SỬ DỤNG (~12–15 trang)
- **2.1. Bài toán phát hiện người và tối ưu suy luận tại biên (Edge AI):**
  - Kiến trúc mạng **YOLOv8s** (Backbone CSPDarknet, Neck PAN-FPN, Anchor-free Detection Head, hàm mất mát CIoU + DFL).
  - Kỹ thuật tối ưu hóa mô hình với **Intel OpenVINO** (biên dịch đồ thị tĩnh IR `.xml/.bin`, lượng tử hóa FP16 kích thước `544×960` giúp giảm thời gian suy luận từ ~200 ms trên CPU xuống ~35–45 ms trên iGPU).
- **2.2. Thuật toán ghép cặp chỗ ngồi và bù xê dịch camera:**
  - Bài toán phân công tối ưu và thuật toán **Hungarian (Kuhn-Munkres)** dựa trên khoảng cách chuẩn hóa theo kích thước ghế.
  - Kỹ thuật ước lượng dịch chuyển toàn cục (**Global Camera Shift Compensation**) bằng trung vị (Median) vector độ lệch tâm ghế:
    $$\Delta \mathbf{c}_{\text{cam}} = \text{median}_{i \in S_{\text{matched}}} (\mathbf{p}_i - \mathbf{c}_i)$$
- **2.3. Ước lượng tư thế đầu 3D (3D Head Pose Estimation) và đặc trưng khuôn mặt:**
  - Mạng **MediaPipe FaceMesh** (478 điểm mốc 3D).
  - Bài toán **Perspective-n-Point (`solvePnP`)** ánh xạ 6 điểm mốc giải phẫu chuẩn (đầu mũi, cằm, khóe mắt trái/phải, khóe miệng trái/phải) từ hệ tọa độ mô hình 3D sang ảnh 2D thông qua ma trận nội tại camera $\mathbf{K}$, chuyển đổi vector quay Rodrigues sang góc Euler (**Yaw, Pitch, Roll**).
  - Công thức chỉ số mở mắt **EAR (Eye Aspect Ratio)** và độ mở miệng **MAR (Mouth Aspect Ratio)**.
- **2.4. Mô hình phân tích chuỗi thời gian hành vi theo chỗ ngồi:**
  - Biểu diễn chuỗi đặc trưng đa phương thức 13 chiều trên cửa sổ trượt 20 mẫu (10 giây ở tần số 2 Hz).
  - Mô hình **SeatScorer (Physically-Constrained Logistic Regression)**: Giải thích tại sao với tập dữ liệu phòng thi nhỏ, việc tổng hợp thống kê trên cửa sổ trượt kết hợp ràng buộc trọng số không âm ($w_j \ge 0$) và chuẩn hóa L2 ($\lambda = 50$) giúp chống quá khớp góc máy tốt hơn mô hình học sâu tự do.
  - Mô hình **Temporal Transformer Encoder** (Positional Encoding + Multi-Head Self-Attention) để đối sánh thực nghiệm.
- **2.5. Kiến trúc phần mềm đa tiến trình (C++ Qt6, ZeroMQ và Protocol Buffers):**
  - Mẫu thiết kế **Publisher–Subscriber** bất đồng bộ qua ZeroMQ và tuần tự hóa nhị phân **Protobuf** giúp tách rời luồng hiển thị 30 FPS (C++) với luồng suy luận 2 Hz (Python).

---

### CHƯƠNG 3: PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG (~18–22 trang)
- **3.1. Phân tích yêu cầu hệ thống:**
  - **Yêu cầu chức năng (Actor: Giám thị coi thi):** Cấu hình phòng thi/camera, Học sơ đồ chỗ ngồi tự động hoặc chỉnh sửa cố định (`seatmap.json`), Giám sát trực tiếp kèm bảng xếp hạng điểm chú ý, Nhận cảnh báo thời gian thực kèm ảnh bằng chứng, Duyệt xác nhận cảnh báo (*Vi phạm thật* / *Báo sai* + Ghi chú SBD), Thống kê trực quan và Xuất biên bản PDF/CSV.
  - **Yêu cầu phi chức năng:** Tốc độ hiển thị video ~30 FPS, nhịp suy luận AI ổn định 2 Hz (độ trễ < 250 ms), tự động phục hồi khi mất kết nối camera/AI, hoạt động hoàn toàn Offline bảo mật dữ liệu học sinh (tuân thủ Nghị định 13/2023/NĐ-CP).
  - **Biểu đồ Use Case tổng quát và đặc tả các Use Case chính.**
- **3.2. Thiết kế kiến trúc tổng thể:**
  - Sơ đồ kiến trúc 3 khối: **Qt6 Desktop Client** ↔ **IPC (ZeroMQ + Protobuf :5555/:5556)** ↔ **Python AI Server** (lấy trực tiếp sơ đồ Mermaid trong `implementation_plan.md`).
  - Thiết kế gói tin giao thức `exam_monitor.proto` (`FramePacket`, `InferenceResult`, `Detection`, `Heartbeat`).
- **3.3. Thiết kế chi tiết Pipeline xử lý AI (`python_ai/`):**
  - Sơ đồ khối luồng xử lý 1 nhịp suy luận (2 Hz): `FramePacket` → `PersonDetector` → `SeatPipeline` (Học ghế 15s / Hungarian Matching / Bù lệch Camera) → `FaceAnalyzer` (PnP Yaw/Pitch/Roll + EAR/MAR) → `FeatureExtractor` (13 chiều × 20 bước) → `BehaviorEngine` (`SeatScorer` + `rule_evaluate` + Bộ lọc chống nhiễu $k\text{-of-}n$) → `ExamSession` (Chụp bằng chứng + Ghi JSON/CSV).
  - Bảng mô tả chi tiết **13 đặc trưng đầu vào** và **7 đặc trưng tổng hợp có ràng buộc vật lý** của `SeatScorer`.
- **3.4. Thiết kế chi tiết Giao diện và Cơ sở dữ liệu tệp (`cpp_client/`):**
  - Sơ đồ lớp C++ (`MainWindow`, `SystemManager`, `VideoThread`, `ZmqPublisher`, `ZmqSubscriber`, `ReportStore`, `DashboardPage`, `LiveViewPage`, `HistoryPage`, `StatsPage`, `SettingsPage`).
  - Cấu trúc lưu trữ dữ liệu phi máy chủ (JSON Schema của `data/reports/YYYY-MM-DD.json`, `seatmap.json`, và cây thư mục bằng chứng `data/evidence/<ngày>/<phòng>/`).

---

### CHƯƠNG 4: HIỆN THỰC HỆ THỐNG VÀ THỰC NGHIỆM ĐÁNH GIÁ (~20–25 trang)
*(Chương quan trọng nhất quyết định điểm A/A+ — Toàn bộ số liệu đã có sẵn trong `python_ai/training/BAO_CAO_DANH_GIA_AI.md`)*
- **4.1. Môi trường cài đặt và hiện thực hóa hệ thống:**
  - Công cụ phát triển: C++20 (MinGW-w64 13.1, Qt 6.11, OpenCV 4.12), Python 3.11 (OpenVINO, MediaPipe, Ultralytics YOLOv8, PyTorch), ZeroMQ & Protobuf.
  - Hình ảnh kết quả hiện thực 5 màn hình giao diện Qt6 và Biên bản PDF A4 xuất ra thực tế.
- **4.2. Xây dựng tập dữ liệu và quy trình khử rò rỉ dữ liệu (Data Deduplication):**
  - Mô tả dữ liệu thu thập: 77 tệp `.npz` (28.398 cửa sổ trượt 10 giây), phát hiện hiện tượng trùng ma trận đặc trưng `X` qua mã băm MD5 và quy trình chuẩn hóa về **12 video gốc độc lập** để đánh giá **Leave-One-Video-Out (LOVO)** trung thực khoa học.
- **4.3. Kết quả đánh giá mô hình AI (Offline Evaluation & Ablation Study):**
  - **Bảng 4.1:** So sánh hiệu năng theo chuẩn Leave-One-Video-Out giữa *Bộ luật cơ sở (`rule`)* (AUC = **0.67**), *Logistic Regression tự do* (AUC = **0.58** — quá khớp góc máy), và *`SeatScorer` đề xuất có ràng buộc vật lý $w \ge 0$* (AUC = **0.75 ± 0.20**).
  - **Bảng 4.2:** Phân tích trọng số vật lý của `SeatScorer` (`pitch_abs_mean4 = +0.479`, `ear_std = +0.435`, `yaw_abs_mean4 = +0.261`...) và 3 mức ngưỡng độ nhạy (`low = 0.906`, `medium = 0.780`, `high = 0.730`).
  - **Bảng 4.3:** Đánh giá tổng thể theo sự kiện (`eval_ai.py --dedup --all-rules`) với các bộ lọc chống nhiễu $k\text{-of-}n$ (giảm tốc độ báo sai từ 64.8 xuống 17.2 và 0.0 báo sai/giờ/chỗ).
  - **Bảng 4.4 (Ablation Study — Bù xê dịch camera):** Khi camera bị rung/lệch 90 px, thuật toán bù dịch chuyển toàn cục giảm số cảnh báo sai từ **8 sự cố xuống 0 sự cố**.
- **4.4. Đánh giá hiệu năng vận hành thời gian thực (System Benchmark):**
  - Tốc độ hiển thị video C++: **~30 FPS (1920×1080 MJPG)**.
  - Thời gian suy luận 1 nhịp AI: **30–45 ms** trên Intel iGPU (OpenVINO FP16) so với **190–220 ms** trên CPU; độ trễ đầu-cuối (End-to-End Latency) từ lúc phát sinh hành vi đến khi hiển thị lên UI đạt **< 250 ms**.
- **4.5. Thực nghiệm theo kịch bản phòng thi mẫu (10 phút) & Quy trình Human-in-the-loop:**
  - Trình bày kết quả chạy kịch bản 6 mốc hành vi (Làm bài bình thường, Quay đầu nhìn bài bạn, Cúi gầm bàn, Rời ghế 15s, Trao đổi đôi, Cầm vật thể) kết hợp giám thị duyệt trên `HistoryPage` và kết xuất Biên bản PDF.

---

### CHƯƠNG 5: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN (~3–4 trang)
- **5.1. Kết luận (Những kết quả đạt được):**
  - Hoàn thiện hệ thống Desktop hoàn chỉnh kết hợp C++ Qt6 và Python OpenVINO qua ZeroMQ/Protobuf.
  - Giải quyết bài toán rò rỉ dữ liệu, đề xuất mô hình `SeatScorer` có ràng buộc vật lý đạt AUC 0.75 và cơ chế bù xê dịch camera chống báo giả.
  - Hiện thực hóa triết lý *"AI gợi ý — Giám thị quyết định"* với quy trình xác minh sự cố và xuất biên bản PDF/CSV tại chỗ.
- **5.2. Hạn chế của đề tài:**
  - Tập dữ liệu huấn luyện thực tế còn nhỏ (12 video gốc độc lập); phạm vi 1 camera bao phủ tốt nhất từ 6–15 thí sinh.
  - Chưa tích hợp mô hình phát hiện vật thể nhỏ (điện thoại, tài liệu phao thi) do đánh đổi tốc độ suy luận thời gian thực trên máy tính cấu hình phổ thông.
- **5.3. Hướng phát triển (Giải quyết câu hỏi Web/Mobile của bạn):**
  - Mở rộng kiến trúc lên **Web Server quản trị tập trung (FastAPI + WebSocket + PostgreSQL)** để kết nối nhiều máy trạm phòng thi về Hội đồng thi.
  - Phát triển **Ứng dụng di động (Mobile App)** nhận thông báo đẩy (Push Notification) kèm ảnh bằng chứng cho giám thị hành lang.
  - Mở rộng tập dữ liệu đa góc máy và tích hợp nhận diện vật thể nhỏ bằng mô hình nhẹ chuyên biệt.

---

## Phần 4. Lịch Trình Hành Động Chi Tiết Trong 14 Ngày (Hoàn Thành Trước Hạn Ở Ngày 11)

| Giai đoạn | Ngày | Công việc cụ thể | Đầu ra (Deliverables) |
|---|---|---|---|
| **Giai đoạn 1: Khóa sổ Phần mềm & Đóng gói Sản phẩm** | **Ngày 1 – 2** (04/11 – 05/11) | 1. Dọn dẹp các file thừa ở thư mục gốc (`models/*.tat`).<br>2. Chạy lại `scripts\package.bat` để cập nhật `dist\ExamMonitorSystem` lên bản mới nhất (02/11).<br>3. Chạy 1 phiên kịch bản mẫu (từ video hoặc webcam) để sinh dữ liệu thật vào `data/reports/`, xác nhận thử trên `HistoryPage` và xuất 1 file **Biên bản PDF + CSV mẫu**. | - Gói sản phẩm `dist/ExamMonitorSystem/` hoàn chỉnh.<br>- Bộ ảnh chụp 5 màn hình UI + 1 file PDF biên bản mẫu. |
| **Giai đoạn 2: Viết Chương 2 & Chương 3** | **Ngày 3 – 6** (06/11 – 09/11) | - **Ngày 3–4:** Viết Chương 2 (YOLOv8s OpenVINO, Hungarian + Bù lệch camera, MediaPipe SolvePnP, SeatScorer $w \ge 0$, ZeroMQ/Protobuf).<br>- **Ngày 5–6:** Viết Chương 3 (Use Case, Kiến trúc 3 khối, Pipeline AI 2 Hz, Thiết kế lớp C++ Qt6 & JSON Schema). | - Bản thảo hoàn chỉnh Chương 2 & Chương 3 (~30–35 trang, đầy đủ hình vẽ & công thức). |
| **Giai đoạn 3: Viết Chương 4 & Chương 5** | **Ngày 7 – 9** (10/11 – 12/11) | - **Ngày 7–8:** Viết Chương 4 (Đưa toàn bộ bảng số liệu từ `BAO_CAO_DANH_GIA_AI.md`, ảnh chụp 5 trang UI, ảnh biên bản PDF A4, bảng đo FPS/Latency).<br>- **Ngày 9:** Viết Chương 5 (Kết luận, Hạn chế, Sơ đồ hướng phát triển Web/Mobile đa phòng thi) + Tóm tắt khóa luận & Mục lục. | - Bản thảo hoàn chỉnh **toàn bộ cuốn Khóa luận (Chương 1 → 5)** vào tối Ngày 9. |
| **Giai đoạn 4: Rà soát, Làm Slide & Tập dượt Bảo vệ** | **Ngày 10 – 11** (13/11 – 14/11) | - **Ngày 10:** Rà soát chính tả, định dạng biểu mẫu IUH, đánh số hình/bảng tự động; quay sẵn **1 video demo dự phòng (2–3 phút)** phòng trường hợp webcam tại phòng hội đồng trục trặc.<br>- **Ngày 11:** Thiết kế Slide thuyết trình (15–18 slides) + Bộ câu hỏi & trả lời phản biện (Q&A). | - **HOÀN THÀNH TRƯỚC HẠN 3 NGÀY!** |
| **Giai đoạn 5: Dự phòng & Nộp quyển** | **Ngày 12 – 14** (15/11 – 17/11) | - Gửi giảng viên hướng dẫn góp ý lần cuối, chỉnh sửa nhẹ, in ấn và tập thuyết trình bấm giờ (10–12 phút). | - Sẵn sàng 100% bảo vệ điểm A/A+. |
