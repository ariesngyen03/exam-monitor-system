"""ai_core.py — Thư viện lõi cho hệ thống giám sát thi cử (BẢN TINH GỌN + TSN).




MỚI — TÍCH HỢP LẠI: nhánh hành vi/tư thế TSN (mmaction2, ResNet50, tự huấn
luyện trên dataset OEP). CHẠY ĐỘC LẬP với Transformer (không ghép chung
vào vector 7 chiều của Transformer để không phải huấn luyện lại) — TSN chỉ
sinh ra 1 tín hiệu bổ trợ "cnn3d_motion_score" (mức độ chuyển động bất
thường) đưa vào rule_based_classify() như 1 lớp kiểm tra thêm, kết hợp
hybrid cùng Transformer trong giamsatthi.py.

File này KHÔNG tự chạy được (không có camera/vòng lặp) — dùng làm thư viện
import cho giamsatthi.py (giám sát chính) / test_cam.py (kiểm thử nhanh).
"""

import csv
from datetime import datetime
import getpass
import json
import math
import os
import platform
import re
import sys
import time
import uuid

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("GLOG_minloglevel", "3")

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist

# Giới hạn số luồng OpenCV để tránh tranh chấp CPU (thread contention) với PyTorch, MediaPipe và giao diện C++
try:
  cv2.setNumThreads(2)
except Exception:
  pass

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)


for _stream in (sys.stdout, sys.stderr):
  try:
    _stream.reconfigure(encoding="utf-8", errors="replace")
  except Exception:
    pass

try:
  import torch
  import torch.nn as nn

  TORCH_AVAILABLE = True
  try:
    _n_With_cores = max(2, min(4, (os.cpu_count() or 4) // 2))
    torch.set_num_threads(_n_With_cores)
  except Exception:
    pass
except ImportError:
  TORCH_AVAILABLE = False
  print("[WARN] PyTorch chưa cài: pip install torch  (dùng rule-based thay thế)")

# Nhánh hành vi/tư thế: TSN (Temporal Segment Network, ResNet50 backbone)
# của mmaction2, đã tự huấn luyện trên dataset OEP (Online Exam
# Proctoring). Chạy ĐỘC LẬP với Transformer — chỉ sinh 1 tín hiệu bổ trợ
# "cnn3d_motion_score" đưa vào rule_based_classify(), KHÔNG ghép vào vector
# đặc trưng của Transformer (giữ nguyên Transformer 7 chiều đang hoạt động
# tốt, không cần huấn luyện lại).
# Pipeline mới (camera trên cao) KHÔNG dùng TSN nữa (thay bằng mức cử động trong features.py) nên mặc định không
# nạp mmaction2 (nạp rất chậm). Muốn dùng lại TSN: đặt biến môi trường AI_CORE_LOAD_TSN=1.
try:
  if os.environ.get("AI_CORE_LOAD_TSN") != "1":
    raise ImportError("TSN không được nạp (AI_CORE_LOAD_TSN != 1)")
  from mmaction.apis import init_recognizer

  MMACTION_AVAILABLE = True
except ImportError:
  MMACTION_AVAILABLE = False


# ==============================================================================
#  1. CẤU HÌNH
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
  # Camera
  CAMERA_ID = 0  # 0 = webcam mặc định, rtsp://... cho IP cam
  FRAME_WIDTH = 1280
  FRAME_HEIGHT = 720
  FPS_TARGET = 30

  # Ngưỡng phát hiện (rule-based / diễn giải feature)
  YAW_THRESHOLD = 15.0  # °: quay đầu sang bên nhìn bài
  PITCH_THRESHOLD = -10.0  # °: cúi xuống xem tài liệu/điện thoại
  EAR_THRESHOLD = 0.25  # Eye Aspect Ratio: liếc mắt mạnh

  # Bền vững theo thời gian trước khi báo CHEATING (chống nhiễu)
  ALERT_SECONDS = 0.67  # ~20 khung hình ở 30fps, tính theo giây thực

  # Face Mesh
  FACE_MESH_EVERY_N_FRAMES = 1  # chạy MediaPipe mỗi N khung hình
  # Độ phân giải đưa vào Face Mesh — TĂNG từ 640 lên 960 để bắt tốt hơn
  # các khuôn mặt NHỎ/XA camera (VD người đứng phía sau lớp học). Đánh
  # đổi: xử lý mỗi khung hình chậm hơn 1 chút (không đáng kể vì Face Mesh
  # vẫn nhẹ hơn nhiều so với TSN/DeepFace — 2 nhánh đó đã chạy nền riêng).
  # Nếu FPS tụt quá thấp trên máy yếu, có thể hạ lại về 640 hoặc 768.
  FACE_PROC_MAX_WIDTH = 768

  # Giám sát ĐA NGƯỜI (cả phòng thi) — mỗi người có trạng thái riêng
  MAX_FACES = 16  # số thí sinh tối đa nhận diện cùng lúc
  TRACK_MAX_DISTANCE_RATIO = 0.15  # ngưỡng khoảng cách gán trùng người (tỉ lệ theo chiều rộng khung hình)
  TRACK_TIMEOUT_FRAMES = (
      20  # số khung hình mất dấu tối đa trước khi bỏ theo dõi 1 người
  )

  # Ngưỡng dò khuôn mặt (BlazeFace bên trong MediaPipe) — hạ thấp hơn mặc
  # định (0.5) để bắt được các góc quay đầu nghiêng mạnh VÀ tư thế cúi
  # đầu viết bài (rất phổ biến khi làm bài thi thật) tốt hơn 1 phần. Hạ
  # quá thấp dễ gây dò nhầm vật thể không phải khuôn mặt (ĐÃ THẤY THỰC TẾ:
  # đầu gối trần bị nhận nhầm thành mặt khi phòng tối + ngưỡng quá thấp).
  # SỬA: nâng nhẹ 0.25 -> 0.32 làm lớp phòng thủ THỨ NHẤT (giảm bớt số lần
  # BlazeFace tự tin nhầm); lớp phòng thủ THỨ HAI là _is_plausible_face()
  # (kiểm tra hợp lý sinh học ngay cả khi BlazeFace vẫn lọt 1 detection sai
  # qua ngưỡng này) — 2 lớp cùng lúc đáng tin hơn hẳn chỉ chỉnh 1 con số.
  FACE_MIN_DETECTION_CONFIDENCE = 0.32
  FACE_MIN_TRACKING_CONFIDENCE = 0.32

  # Nếu 1 người ĐANG được theo dõi (đã xuất hiện ổn định) bỗng dưng MẤT
  # DẤU hoàn toàn (model không dò được mặt — thường do quay đầu quá
  # nghiêng, cúi hẳn xuống, hoặc rời khỏi khung hình) trong >= số giây
  # này, hệ thống coi đây là 1 DẤU HIỆU ĐÁNG NGỜ (thay vì im lặng bỏ qua).
  FACE_LOST_ALERT_SECONDS = 2.0

  # Transformer / chuỗi đặc trưng
  SEQ_LEN = 20          # 20 mẫu x 0.5s = cửa sổ 10 giây (FEATURE_HZ = 2)
  FEATURE_HZ = 2.0      # mẫu đặc trưng / giây / chỗ ngồi — PHẢI giống nhau lúc huấn luyện và lúc chạy
  # NỢ KỸ THUẬT (chưa xử lý trong bản này — cần huấn luyện lại model để
  # giải quyết dứt điểm): model Transformer hiện có được huấn luyện với
  # vector 22 chiều CŨ (gồm cả cảm xúc/thiết bị của pipeline trước), còn
  # extract_features() bây giờ chỉ điền 7 tín hiệu thật rồi PAD 0 phần
  # còn lại -> lệch phân bố dữ liệu train/inference, Transformer có thể
  # đánh giá thấp hơn thực tế. Đây là lý do rule_based_classify() LUÔN
  # chạy song song bắt buộc (không phải tùy chọn) — xem giamsatthi.py.
  FEATURE_DIM = 13  # xem features.py; model cũ 22 chiều KHÔNG dùng được, cần huấn luyện lại (train_transformer.py)
  # Đường dẫn TUYỆT ĐỐI tới model Transformer đã huấn luyện — luôn trỏ vào
  # thư mục "models/" nằm CẠNH ai_core.py, bất kể chạy lệnh từ đâu.
  # VD: nếu ai_core.py nằm ở C:\...\ai_issue_classifier\ai_core.py thì
  # MODEL_PATH sẽ tự trỏ đúng C:\...\ai_issue_classifier\models\transformer_model.pt
  MODEL_PATH = os.path.join(BASE_DIR, "models", "transformer_seat.pt")
  MODEL_META_PATH = os.path.join(BASE_DIR, "models", "transformer_seat_meta.json")
  CHEAT_THRESHOLD = 0.60  # ngưỡng mặc định; nếu có transformer_seat_meta.json thì dùng ngưỡng chọn lúc huấn luyện
  # Hiển thị số % cụ thể trên HUD hay không — mặc định TẮT (theo yêu cầu
  # thực tế: % là điểm rủi ro heuristic, không phải xác suất thống kê thật,
  # hiển thị số cụ thể dễ gây hiểu nhầm về độ chính xác). Khi tắt, HUD chỉ
  # hiện mức độ định tính (THẤP/TRUNG BÌNH/CAO/RẤT CAO). Số % vẫn được ghi
  # đầy đủ vào CSV/JSON để tham khảo/phân tích sau, chỉ ẩn trên màn hình live.
  SHOW_FRAUD_PERCENT = False
  EMA_ALPHA = 0.35  # làm mịn xác suất theo thời gian (EMA)
  # TỐI ƯU MÁY YẾU: chỉ THỰC SỰ suy luận Transformer 1 lần mỗi
  # TRANSFORMER_INFER_SKIP khung hình (2 = cách khung hình, 1 = mỗi khung
  # hình như trước). Giảm tải CPU đáng kể trên máy cấu hình thấp, gần
  # như không ảnh hưởng độ nhạy thực tế (hành vi đầu người không đổi đủ
  # nhanh để cần suy luận lại mỗi khung hình liên tục).
  TRANSFORMER_INFER_SKIP = 2
  # Chế độ kết hợp Transformer + rule-based — xem giải thích chi tiết tại
  # nơi dùng trong giamsatthi.py. "or" = ưu tiên không bỏ sót (mặc định,
  # hành vi cũ); "and" = ưu tiên giảm báo động giả, chấp nhận rủi ro bỏ sót.
  # "and": Transformer chỉ có thể XÁC NHẬN THÊM cho luật hành vi, không tự tạo báo động — an toàn khi
  # model chưa được kiểm chứng đủ tốt trên dữ liệu chưa rò rỉ (xem model_seat_meta.json: val_precision thấp).
  # Đổi lại "or" khi model đã có precision/recall ổn định qua --split video với nhiều video đa dạng hơn.
  HYBRID_FUSION_MODE = "and"

  # ── TSN (mmaction2) — nhánh hành vi/tư thế ĐỘC LẬP, bổ trợ cho
  # rule_based_classify() qua tín hiệu "cnn3d_motion_score" (KHÔNG ghép
  # vào vector 7 chiều của Transformer) ───────────────────────────────
  CNN3D_CLIP_LEN = 40  # số khung hình camera thô giữ trong buffer/người (cửa sổ lấy mẫu)
  TSN_NUM_SEGMENTS = 8  # số khung hình lấy mẫu rải đều từ buffer, PHẢI khớp lúc huấn luyện TSN
  CNN3D_FEATURE_DIM = (
      8  # số chiều đặc trưng TSN sau khi giảm chiều (Linear+Tanh)
  )
  CNN3D_PROC_SIZE = (
      224  # kích thước crop cuối cùng đưa vào TSN (chuẩn ImageNet/ResNet)
  )
  CNN3D_RESIZE_SHORT_SIDE = (
      256  # resize cạnh ngắn về giá trị này TRƯỚC KHI crop 224
  )
  CNN3D_BUFFER_WIDTH = (
      320  # chiều rộng lưu trong buffer/người (giữ tỉ lệ gốc, nhẹ bộ nhớ)
  )
  CNN3D_MOTION_THRESHOLD = 0.5  # ngưỡng L2-norm giữa 2 clip liên tiếp để coi là "chuyển động bất thường"
  # Hệ số làm mượt (EMA) bbox TRƯỚC KHI cắt ảnh gửi cho TSN — chống "rung
  # giả" (xem giải thích tại nơi dùng trong giamsatthi.py). Nhỏ hơn = mượt
  # hơn nhưng phản ứng chậm hơn với chuyển động thật; 0.4 là điểm cân bằng.
  TSN_CROP_SMOOTH_ALPHA = 0.4
  # Trọng số cnn3d_motion_score khi dùng làm tín hiệu DUY NHẤT (người
  # KHÔNG thấy mặt — xem rule_based_classify(face_visible=False)) — phải
  # >= 5 (ngưỡng CHEATING) để tự nó đủ kết luận, KHÁC với trọng số bổ trợ
  # rất thấp (+3, cộng thêm bên cạnh yaw/pitch/EAR/gaze) khi có mặt.
  CNN3D_MOTION_SCORE_WEIGHT_NO_FACE = 8

  # ── Chống nhiễu (debounce/hysteresis) cho rule_based_classify() ─────
  # Trước đây rb_label được dùng TRỰC TIẾP mỗi khung hình — nhưng góc
  # yaw/pitch đo qua solvePnP luôn rung nhẹ dù đầu gần như đứng yên, nên
  # chỉ cần dao động quanh ngưỡng là rb_label (và theo đó là final_label)
  # nhảy qua lại NORMAL/CHEATING MỖI KHUNG HÌNH -> % gian lận hiển thị
  # nhảy loạn. Yêu cầu rb_label phải LIÊN TỤC đủ số khung hình mới thực
  # sự đổi trạng thái (vào nhanh, ra chậm hơn 1 chút để không "nháy").
  RB_LABEL_ENTER_FRAMES = 3   # số khung hình LIÊN TỤC báo CHEATING mới thực sự tính là CHEATING
  RB_LABEL_EXIT_FRAMES = 5    # số khung hình LIÊN TỤC báo NORMAL mới thực sự thoát CHEATING
  # Làm mượt % gian lận HIỂN THỊ (EMA) — tách biệt với EMA_ALPHA của
  # Transformer, chỉ áp dụng cho con số hiển thị trên HUD, không ảnh
  # hưởng tới quyết định CHEATING/NORMAL thật (đã được debounce ở trên).
  FRAUD_PCT_EMA_ALPHA = 0.3
  # Đường dẫn TUYỆT ĐỐI tới config + checkpoint TSN đã tự huấn luyện —
  # luôn trỏ vào thư mục mmaction2/ nằm CẠNH ai_core.py.
  MMACTION_CONFIG = os.path.join(
      BASE_DIR,
      "mmaction2",
      "configs",
      "recognition",
      "tsn",
      "tsn_r50_1x1x8_100e_oep-rgb.py",
  )
  MMACTION_CHECKPOINT = os.path.join(
      BASE_DIR,
      "mmaction2",
      "work_dirs",
      "tsn_r50_oep",
      "best_acc_top1_epoch_1.pth",
  )
  MMACTION_DEVICE = "cpu"  # "cpu" hoặc "cuda:0" nếu có GPU
  MMACTION_BACKBONE_FEATURE_DIM = 2048  # khớp in_channels=2048 trong TSNHead
  MMACTION_NORM_MEAN = [123.675, 116.28, 103.53]
  MMACTION_NORM_STD = [58.395, 57.12, 57.375]
  # Đặt False để tắt hoàn toàn cảnh báo thiếu checkpoint và tiết kiệm CPU khi chưa train model OEP
  TSN_ENABLED = False  # pipeline camera trên cao không dùng TSN (chưa có checkpoint); mức cử động thay thế

  # ── Cảm xúc (DeepFace) — nhánh ĐỘC LẬP, chạy nền như 3DCNN, bổ trợ cho
  # rule_based_classify() qua cờ "cảm xúc lo lắng/căng thẳng kéo dài" ────
  EMOTION_ENABLED = True
  EMOTION_SLIDING_WINDOW_SIZE = (
      7  # số khung hình gần nhất dùng làm mượt (statistics.mode)
  )
  # Các nhãn DeepFace trả về coi là dấu hiệu "lo lắng/căng thẳng":
  # angry, disgust, fear, happy, neutral, sad, surprise — chọn 3 nhãn có
  # liên hệ tới trạng thái căng thẳng/lo âu khi bị phát hiện gian lận.
  EMOTION_STRESS_LABELS = ["fear", "sad", "angry"]
  EMOTION_STRESS_SUSTAIN_SECONDS = (
      3.0  # phải kéo dài liên tục bấy lâu mới tính là đáng ngờ
  )
  FACE_MARGIN_RATIO = 0.1  # mở rộng khung mặt khi cắt cho DeepFace (giữ đúng tỉ lệ bản gốc AI_test)
  # Độ sáng trung bình TỐI THIỂU (thang 0-255) của ảnh khuôn mặt để DeepFace
  # được coi là đủ tin cậy để phân tích — dưới ngưỡng này, trả về nhãn
  # "khong ro (anh toi)" thay vì đoán bừa 1 cảm xúc cụ thể trông đáng tin
  # nhưng thực ra không có cơ sở (ảnh quá tối).
  EMOTION_MIN_BRIGHTNESS = 40
  EMOTION_HOLD_SECONDS = 5.0  # cảm xúc mới phải ổn định liên tục đủ lâu mới đổi hiển thị, tránh nhảy qua lại

  # ── Học vị trí chỗ ngồi TỰ ĐỘNG (không cần giám thị hiệu chỉnh tay) ──
  SEAT_LEARNING_ENABLED = True
  SEAT_MIN_SAMPLES = 10  # số khung hình tối thiểu 1 người phải xuất hiện mới tính là 1 "chỗ ngồi" thật (tránh người đi ngang thoáng qua)
  SEAT_ROW_GAP_RATIO = 0.12  # khoảng cách Y tối thiểu (tỉ lệ theo chiều cao khung hình) để tách thành 1 hàng mới
  SEAT_MATCH_MAX_DIST_RATIO = 0.20  # khoảng cách tối đa (tỉ lệ theo chiều rộng khung hình) để coi 1 vị trí hiện tại là "khớp" với 1 chỗ đã học
  # SỬA (chống học sai chỗ ngồi): độ lệch chuẩn TỐI ĐA (tỉ lệ theo chiều
  # rộng khung hình) của các mẫu vị trí 1 track trong lúc học — vượt quá
  # ngưỡng này nghĩa là track đó khả năng cao là NGƯỜI ĐI NGANG QUA nhiều
  # chỗ khác nhau (không ngồi yên 1 chỗ thật), bị LOẠI khỏi danh sách chỗ
  # ngồi thay vì học nhầm thành 1 "chỗ" nằm giữa đường đi của họ.
  SEAT_POSITION_MAX_STD_RATIO = 0.06

  # ── PIPELINE CAMERA TRÊN CAO (seat_map / person_detector / face_analyzer / behavior) ──
  PERSON_BACKEND = "auto"            # "auto" | "yolo" | "torchvision"
  # yolov8s thay yolov8n (19/10/2026): camera đặt cao nhìn chéo, người ngồi sau bị che -> bản n mất người ngồi sau
  # 13–27% số khung, bản s 6–7%; toàn bộ AI ~245 ms/nhịp trên CPU (vẫn đủ 2 nhịp/giây). Máy yếu có thể đổi lại yolov8n.pt
  PERSON_YOLO_MODEL = "yolov8s.pt"
  PERSON_IMGSZ = 960                 # tăng 1280 nếu bỏ sót người nhỏ, đổi lại chậm hơn
  # Cùng yolov8s xuất sang OpenVINO (khung 544x960 = 1080p thu nhỏ còn 960 như PERSON_IMGSZ), dùng tự động khi có thư
  # mục này + `pip install openvino`. Đo trên i5-12500H: GPU Iris Xe ~30 ms, CPU ~190 ms (PyTorch ~190 ms), box trùng khớp.
  # Tạo lại: python python_ai/tools/export_openvino.py
  PERSON_OPENVINO_MODEL = os.path.join(BASE_DIR, "models", "yolov8s_openvino_544x960", "yolov8s.xml")
  PERSON_DEVICE = "auto"             # "auto" = GPU tích hợp Intel nếu có, không thì CPU | "gpu" | "cpu"
  PERSON_THREADS = 4                 # số luồng khi chạy trên CPU — chừa phần còn lại cho camera + giao diện C++
  PERSON_CONF = 0.35
  PERSON_MIN_AREA_RATIO = 0.002
  PERSON_MAX_AREA_RATIO = 0.25
  SEAT_CALIBRATION_SECONDS = 15.0    # thời gian học chỗ ngồi đầu phiên (mọi người nên đã ngồi yên)
  SEAT_COL_RIGHT_TO_LEFT = True       # True: "Cho 1" ở bên PHẢI khung hình, đánh số dần sang trái
  # Phòng có bàn xếp thành DÃY KÉO DÀI THEO CHIỀU SÂU camera (VD 2 dãy trái/phải, mỗi dãy nhiều bàn nối
  # tiếp vào xa — đúng cấu trúc phòng đã xác nhận qua video thật) cần True: "Dãy" nhóm theo trái/phải,
  # "Ghế" đánh số theo độ sâu. Phòng xếp nhiều hàng ngang cách đều camera thì để False (Dãy theo độ sâu).
  SEAT_ROW_BY_X = True
  SEAT_FAR_FIRST = False              # chỉ dùng khi SEAT_ROW_BY_X=True: True = "Ghe 1" la ban XA camera nhat
  # Tự động học lại TOÀN BỘ chỗ ngồi khi người mới (không khớp chỗ đã học) đứng/ngồi ổn định đủ lâu.
  # CẢNH BÁO: học lại xoá cả tiến trình theo dõi của những người ĐANG ngồi ổn định (thời gian nghi ngờ,
  # thời gian vắng mặt...), không chỉ riêng người mới — đã xác nhận với người dùng, đây là hành vi mong muốn.
  SEAT_NEWCOMER_RELEARN_ENABLED = True
  SEAT_NEWCOMER_RELEARN_SECONDS = 8.0    # phải ổn định liên tục ít nhất chừng này giây mới kích hoạt học lại
  SEAT_NEWCOMER_STABLE_RADIUS_W = 0.6    # bán kính (theo bề rộng người) coi là "cùng 1 người" giữa các lần phân tích
  SEAT_MATCH_RADIUS_W = 1.5          # khoảng cách tối đa (theo bề rộng người) để coi là còn ở chỗ đó
  SEAT_STICKY_DISCOUNT = 0.7         # <1.0: ưu tiên giữ nguyên ghép của nhịp trước, chống nhấp nháy giữa 2 chỗ gần nhau
  FACE_HEAD_FRAC = 0.45              # vùng đầu = 45% chiều cao box từ đỉnh xuống
  FACE_CROP_TARGET_SIDE = 320        # phóng vùng đầu tới cạnh dài ~320px trước khi đưa vào Face Mesh
  FACE_HOLD_SECONDS = 0.6            # giữ kết quả mặt ngắn hạn để chống nhấp nháy
  # Luật hành vi: giá trị khởi đầu, CHƯA kiểm chứng trên dữ liệu thật của bạn — chỉnh sau khi chạy thử
  SEAT_YAW_THRESHOLD = 30.0
  SEAT_PITCH_THRESHOLD = 25.0
  SEAT_LEAN_MILD = 0.7
  SEAT_LEAN_STRONG = 1.2
  SEAT_SIZE_CHANGE = 0.35
  SEAT_MOTION_HIGH = 0.6
  SEAT_ABSENT_ALERT_SECONDS = 300.0   # 5 phút — rời chỗ lâu hơn mức này mới cảnh báo
  # Chống nhiễu bất đối xứng: CHEATING báo gần như ngay (0s = 1 nhịp), NORMAL cần giữ ổn định đủ lâu mới
  # hiển thị lại (mặc định 5 giây) — tránh chớp tắt trạng thái khi hành vi dao động quanh ngưỡng.
  SEAT_ENTER_SECONDS = 0.0
  SEAT_EXIT_SECONDS = 5.0
  SEAT_PROB_EMA = 0.5

  # ── (ĐÃ BỎ theo yêu cầu) Trước đây có nhánh dò NGƯỜI toàn thân bằng
  # Faster R-CNN (person_detector.py, PersonDetectorWorker) để bổ trợ khi
  # Face Mesh không thấy mặt. Đã loại bỏ hoàn toàn do liên tục gây báo
  # động giả (dò nhầm vật thể, đếm trùng người, dò nhầm đầu gối...) khi
  # kiểm thử thực tế với camera đặt trên cao. Hệ thống giờ CHỈ dựa trên
  # khuôn mặt (Face Mesh) — người hoàn toàn không thấy mặt (quay lưng/che
  # mặt/rời chỗ) vẫn được phát hiện qua cơ chế "mất dấu" hiện có trong
  # giamsatthi.py (dựa trên SimpleFaceTracker, không cần model bổ sung).

  # Logging & bằng chứng — cũng dùng đường dẫn TUYỆT ĐỐI theo BASE_DIR để
  # log/ảnh/báo cáo luôn nằm đúng 1 chỗ (cạnh ai_core.py), dù chạy lệnh
  # từ thư mục nào (Desktop, cmd, app_giamsat.py subprocess...).
  LOG_FILE = os.path.join(BASE_DIR, "cheating_log.csv")
  EVIDENCE_DIR = os.path.join(
      BASE_DIR, "evidence"
  )  # ảnh lưu vào evidence/YYYY-MM-DD/<phòng>/
  STOP_SIGNAL_FILE = os.path.join(BASE_DIR, "stop_signal.tmp")
  FRAUD_CAPTURE_THRESHOLD = 70.0  # % gian lận để tự động chụp bằng chứng
  SHOW_EVIDENCE_POPUP_WINDOW = False

  # Hiển thị (HUD)
  SHOW_LANDMARKS = True
  FONT = cv2.FONT_HERSHEY_SIMPLEX

  # Báo cáo JSON
  JSON_REPORT_FILE = os.path.join(BASE_DIR, "session_reports.json")
  JSON_REPORT_DIR = os.path.join(BASE_DIR, "reports")
  EXAM_ROOM_ID = "PHONG_THI_01"
  PROCTOR_NAME = ""  # "" = lấy từ username hệ thống


# ==============================================================================
#  2. MEDIAPIPE — FACE MESH (Yaw/Pitch/Roll, EAR, Gaze)
# ==============================================================================

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

MODEL_3D_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype=np.float64,
)

MP_6PT_IDX = [1, 152, 263, 33, 287, 57]
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
# Điểm mống mắt (iris) — CHỈ có khi FaceMesh khởi tạo với refine_landmarks=
# True (478 điểm thay vì 468 điểm mặc định). Chính xác hơn NHIỀU so với
# cách cũ (nhị phân hóa ảnh xám theo ngưỡng độ sáng cố định — rất dễ sai
# khi ánh sáng phòng thi thay đổi, da/mắt màu tối, hoặc đeo kính).
LEFT_IRIS_IDX = [474, 475, 476, 477]
RIGHT_IRIS_IDX = [469, 470, 471, 472]


def compute_head_pose(landmarks, img_w, img_h):
  """Tính góc Yaw, Pitch, Roll từ 6 điểm landmark qua solvePnP."""
  fl = img_w
  cam = np.array(
      [[fl, 0, img_w / 2], [0, fl, img_h / 2], [0, 0, 1]], dtype=np.float64
  )
  pts = np.array(
      [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in MP_6PT_IDX],
      dtype=np.float64,
  )
  ok, rvec, _ = cv2.solvePnP(
      MODEL_3D_POINTS,
      pts,
      cam,
      np.zeros((4, 1)),
      flags=cv2.SOLVEPNP_ITERATIVE,
  )
  if not ok:
    return 0.0, 0.0, 0.0
  rmat, _ = cv2.Rodrigues(rvec)
  angles, *_ = cv2.RQDecomp3x3(rmat)
  yaw, pitch, roll = angles[1], angles[0], angles[2]
  if not (-100 <= yaw <= 100 and -100 <= pitch <= 100 and -100 <= roll <= 100):
    return 0.0, 0.0, 0.0  # chặn giá trị rác do lỗi tracking
  return yaw, pitch, roll


def compute_ear(landmarks, eye_indices, img_w, img_h):
  """Eye Aspect Ratio — EAR thấp → liếc/nhắm mắt."""
  pts = np.array(
      [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in eye_indices]
  )
  A = dist.euclidean(pts[1], pts[5])
  B = dist.euclidean(pts[2], pts[4])
  C = dist.euclidean(pts[0], pts[3])
  return (A + B) / (2.0 * C + 1e-6)


def compute_gaze_ratio(landmarks, eye_indices, img_w, img_h, gray):
  """Tỷ lệ vị trí đồng tử trong vùng mắt (ROI cục bộ để tối ưu tốc độ).

  KHUYẾT ĐIỂM CỦA CÁCH LÀM CŨ (giữ lại hàm này làm fallback — xem
  compute_gaze_ratio_iris() bên dưới cho cách chính xác hơn): nhị phân hóa
  ảnh xám theo 1 ngưỡng ĐỘ SÁNG CỐ ĐỊNH (threshold=70) để "đoán" đồng tử là
  vùng tối nhất trong mắt — RẤT DỄ SAI khi: ánh sáng phòng thi thay đổi
  (đèn, cửa sổ), da/mắt màu tối làm cả vùng mắt đều tối, hoặc đeo kính
  (phản chiếu ánh sáng tạo vùng tối giả). Đây là hạn chế đã biết, KHÔNG
  sửa được triệt để chỉ bằng chỉnh ngưỡng — cần dữ liệu mống mắt thật.
  """
  pts = np.array(
      [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h))
       for i in eye_indices]
  )
  x0, y0 = pts[:, 0].min(), pts[:, 1].min()
  x1, y1 = pts[:, 0].max(), pts[:, 1].max()
  if x1 <= x0 or y1 <= y0:
    return 0.5
  roi = gray[y0:y1, x0:x1]
  if roi.size == 0:
    return 0.5
  lp = pts.copy()
  lp[:, 0] -= x0
  lp[:, 1] -= y0
  mask = np.zeros(roi.shape, np.uint8)
  cv2.fillConvexPoly(mask, cv2.convexHull(lp), 255)
  _, th = cv2.threshold(roi, 70, 255, cv2.THRESH_BINARY_INV)
  th = cv2.bitwise_and(th, mask)
  _, ww = th.shape
  if ww == 0:
    return 0.5
  L = np.sum(th[:, : ww // 2] == 255)
  R = np.sum(th[:, ww // 2 :] == 255)
  return R / (L + R + 1e-6)


def compute_gaze_ratio_iris(landmarks, eye_indices, iris_indices, img_w, img_h):
  """Tỷ lệ vị trí đồng tử — CÁCH CHÍNH XÁC HƠN, dùng TRỰC TIẾP tọa độ điểm

  mống mắt (iris) mà MediaPipe đã tự dò riêng (khi refine_landmarks=True),
  thay vì đoán qua ngưỡng độ sáng ảnh xám. Không phụ thuộc ánh sáng/màu
  da/kính vì MediaPipe đã tự học nhận diện mống mắt từ dữ liệu huấn luyện
  đa dạng — đây là fix thật cho hạn chế đã nêu ở compute_gaze_ratio() cũ.

  Trả về tỷ lệ [0,1]: vị trí tâm mống mắt theo trục ngang, chuẩn hóa theo
  chiều rộng mắt (0=sát góc trái, 1=sát góc phải, ~0.5=nhìn thẳng) — giữ
  ĐÚNG Ý NGHĨA và khoảng giá trị như hàm cũ để tương thích ngược với
  GAZE_THRESHOLD hiện có trong Config, không cần đổi ngưỡng.
  """
  eye_pts = np.array(
      [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in eye_indices]
  )
  x0, x1 = eye_pts[:, 0].min(), eye_pts[:, 0].max()
  if x1 <= x0:
    return 0.5
  iris_cx = np.mean([landmarks[i].x * img_w for i in iris_indices])
  return float(np.clip((iris_cx - x0) / (x1 - x0 + 1e-6), 0.0, 1.0))


def init_face_mesh(max_faces=None):
  """Khởi tạo đối tượng MediaPipe FaceMesh — nhận diện TỐI ĐA Config.MAX_FACES

  khuôn mặt cùng lúc trong 1 khung hình (giám sát cả phòng thi).

  LƯU Ý: MediaPipe FaceMesh (BlazeFace) được huấn luyện chủ yếu cho khuôn
  mặt gần chính diện — khi quay đầu >~60-70° (gần như nhìn nghiêng hẳn),
  model có thể KHÔNG dò được mặt nào cả (khác với việc đo yaw/pitch sai).
  Giảm min_detection_confidence giúp bắt được góc nghiêng tốt hơn 1 phần,
  nhưng KHÔNG loại bỏ hoàn toàn giới hạn này — xem thêm cơ chế phát hiện
  "mất dấu đột ngột" trong giamsatthi.py để bù cho trường hợp dò trượt.
  """
  n = max_faces or getattr(Config, "MAX_FACES", 16)
  return mp_face_mesh.FaceMesh(
      max_num_faces=n,
      # SỬA: BẬT refine_landmarks — trước đây tắt (False), khiến hệ thống
      # PHẢI dùng cách đoán gaze thô qua ngưỡng độ sáng ảnh xám (dễ sai).
      # Bật lên cho MediaPipe tự dò thêm 10 điểm mống mắt (iris), CHỈ tốn
      # thêm chi phí tính toán không đáng kể (vẫn cùng 1 model FaceMesh),
      # đổi lại gaze chính xác hơn hẳn — xem compute_gaze_ratio_iris().
      refine_landmarks=True,
      min_detection_confidence=getattr(
          Config, "FACE_MIN_DETECTION_CONFIDENCE", 0.35
      ),
      min_tracking_confidence=getattr(
          Config, "FACE_MIN_TRACKING_CONFIDENCE", 0.35
      ),
  )


FACE_OVAL_IDX = [
    10,
    109,
    67,
    103,
    54,
    21,
    162,
    127,
    234,
    93,
    132,
    58,
    172,
    136,
    150,
    149,
    176,
    148,
    152,
]


def _is_plausible_face(lms, x0, y0, x1, y1):
  """Kiểm tra hợp lý sinh học (anatomical sanity check) — SỬA lỗi thật đã

  gặp qua ảnh thực tế: MediaPipe FaceMesh (BlazeFace) đôi khi dò NHẦM 1
  vùng da không phải mặt (VD đầu gối trần, cánh tay) thành "khuôn mặt" —
  đặc biệt dễ xảy ra khi FACE_MIN_DETECTION_CONFIDENCE bị hạ thấp (để bắt
  góc nghiêng tốt hơn) kết hợp ánh sáng yếu (chi tiết da mờ, dễ gây nhầm
  lẫn). Faces landmarks vẫn được trả về (68/478 điểm hợp lệ về mặt toán
  học), nhưng KHÔNG có ý nghĩa giải phẫu thật — hàm này kiểm tra nhanh,
  rẻ (không cần model nào thêm) để LOẠI BỎ các trường hợp vô lý đó:

  1. Thứ tự trên-dưới: trán (landmark 10) phải ở TRÊN mũi (landmark 1),
     mũi phải ở TRÊN cằm (landmark 152) — 1 khuôn mặt thật luôn đúng thứ
     tự này; 1 vùng da ngẫu nhiên (như đầu gối) thường KHÔNG tuân theo.
  2. Mắt phải ở TRÊN mũi — dùng góc mắt landmark 33 (mắt phải).
  3. Tỉ lệ khung bao (width/height) phải nằm trong khoảng hợp lý của
     khuôn mặt người thật (khoảng 0.5-1.15) — đầu gối/cánh tay thường có
     tỉ lệ khác hẳn (dẹt/dài hơn nhiều).

  Trả về True nếu HỢP LÝ (giữ lại), False nếu VÔ LÝ (loại bỏ, coi như
  không dò được mặt ở vị trí này)."""
  h_box = max(1.0, y1 - y0)
  w_box = max(1.0, x1 - x0)
  aspect = w_box / h_box
  if not (0.5 <= aspect <= 1.15):
    return False

  forehead_y = lms[10].y
  nose_y = lms[1].y
  chin_y = lms[152].y
  eye_y = lms[33].y
  face_h = chin_y - forehead_y
  if face_h <= 0:
    return False
  min_gap = 0.08 * face_h
  if not (forehead_y + min_gap < nose_y < chin_y - min_gap):
    return False
  if not (eye_y < nose_y):
    return False
  return True


def process_faces_frame(face_mesh, frame, gray, rgb, cfg=Config):
  """Chạy MediaPipe Face Mesh trên 1 khung hình, vẽ viền mặt + mũi tên hướng

  đầu lên `frame` cho TỪNG khuôn mặt phát hiện được, trả về DANH SÁCH
  (list) tất cả khuôn mặt — dùng cho giám sát nhiều thí sinh cùng lúc.

  Mỗi phần tử trong danh sách trả về là 1 dict:
      {yaw, pitch, roll, ear_l, ear_r, gaze_r, cx, cy, bbox}
  trong đó (cx, cy) là tâm khuôn mặt (để gán ID theo dõi — tracking) và
  bbox = (x0, y0, x1, y1) là khung bao quanh mặt (để vẽ nhãn trạng thái).
  """
  h, w = frame.shape[:2]
  max_w = getattr(cfg, "FACE_PROC_MAX_WIDTH", 640)
  if w > max_w:
    s = max_w / float(w)
    rgb_proc = cv2.resize(
        rgb, (max_w, max(1, int(h * s))), interpolation=cv2.INTER_LINEAR
    )
  else:
    rgb_proc = rgb

  fres = face_mesh.process(rgb_proc)
  faces = []
  if not fres.multi_face_landmarks:
    return faces

  show_landmarks = getattr(cfg, "SHOW_LANDMARKS", True)
  for fl in fres.multi_face_landmarks:
    lms = fl.landmark
    xs = [p.x * w for p in lms]
    ys = [p.y * h for p in lms]
    x0, x1 = int(min(xs)), int(max(xs))
    y0, y1 = int(min(ys)), int(max(ys))

    # SỬA (lọc "khuôn mặt" vô lý — VD đầu gối/cánh tay bị nhận nhầm, xem
    # giải thích đầy đủ ở _is_plausible_face()): loại NGAY trước khi vẽ
    # hoặc tính bất kỳ đặc trưng nào, coi như không dò được mặt ở đây.
    if not _is_plausible_face(lms, x0, y0, x1, y1):
      continue

    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0

    if show_landmarks:
      ov_pts = np.array(
          [(int(lms[i].x * w), int(lms[i].y * h)) for i in FACE_OVAL_IDX],
          dtype=np.int32,
      )
      cv2.polylines(frame, [ov_pts], True, (0, 200, 120), 1)

    yaw, pitch, roll = compute_head_pose(lms, w, h)
    ear_l = compute_ear(lms, LEFT_EYE_IDX, w, h)
    ear_r = compute_ear(lms, RIGHT_EYE_IDX, w, h)
    # Dùng gaze qua IRIS (chính xác hơn) nếu FaceMesh đã bật refine_landmarks
    # (478 điểm — có đủ 10 điểm mống mắt); tự động rơi về cách cũ (ngưỡng
    # ảnh xám) nếu vì lý do gì đó chỉ có 468 điểm — không bao giờ crash.
    if len(lms) >= 478:
      gaze_r = compute_gaze_ratio_iris(lms, RIGHT_EYE_IDX, RIGHT_IRIS_IDX, w, h)
    else:
      gaze_r = compute_gaze_ratio(lms, LEFT_EYE_IDX, w, h, gray)

    n2d = (int(lms[1].x * w), int(lms[1].y * h))
    cv2.arrowedLine(
        frame,
        n2d,
        (
            n2d[0] + int(math.sin(math.radians(yaw)) * 60),
            n2d[1] + int(math.sin(math.radians(pitch)) * 60),
        ),
        (0, 180, 255),
        2,
        tipLength=0.3,
    )

    faces.append({
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
        "ear_l": ear_l,
        "ear_r": ear_r,
        "gaze_r": gaze_r,
        "cx": cx,
        "cy": cy,
        "bbox": (x0, y0, x1, y1),
    })
  return faces


class SimpleFaceTracker:
  """Bộ theo dõi (tracking) ĐƠN GIẢN, gán 1 ID cố định cho từng thí sinh

  xuyên suốt phiên giám sát — dựa trên khoảng cách tâm khuôn mặt (centroid)
  giữa 2 khung hình liên tiếp (không dùng thư viện tracking ngoài, đủ nhẹ
  và đủ ổn định cho bối cảnh phòng thi: người ngồi yên, ít di chuyển mạnh).

  Cách hoạt động mỗi khung hình:
    1. Với mỗi khuôn mặt mới phát hiện, tìm track (người) cũ gần nhất
       (khoảng cách tâm < ngưỡng) → gán lại đúng ID người đó.
    2. Khuôn mặt không khớp với track nào → cấp ID MỚI (người mới xuất hiện).
    3. Track cũ không khớp khuôn mặt nào trong nhiều khung hình liên tiếp
       (VD do quay đầu mất dấu tạm thời) → tăng bộ đếm "lost"; quá
       TRACK_TIMEOUT_FRAMES thì mới xoá hẳn (tránh đổi ID chỉ vì mất dấu 1-2
       khung hình).
  """

  def __init__(self, max_distance_ratio=None, timeout_frames=None):
    self.max_distance_ratio = (
        max_distance_ratio or Config.TRACK_MAX_DISTANCE_RATIO
    )
    self.timeout_frames = timeout_frames or Config.TRACK_TIMEOUT_FRAMES
    self.next_id = 1
    self.tracks = {}  # track_id -> {"cx", "cy", "lost"}

  def update(self, detections, frame_w):
    """detections: list[dict] từ process_faces_frame().

    Trả về dict: {track_id: detection_dict} đã gán ID ổn định.

    SỬA: trước đây gán TỪNG khuôn mặt theo kiểu THAM LAM (lần lượt xét
    từng detection, chọn track gần nhất còn trống) — với lớp đông người
    ngồi sát nhau, thứ tự xét ảnh hưởng tới kết quả và dễ gây HOÁN ĐỔI ID
    giữa 2 người khi họ di chuyển gần đồng thời (VD detection A đáng lẽ
    khớp track 2 nhưng bị detection B "giành" trước vì được xét trước).
    Giờ dùng thuật toán ghép cặp TỐI ƯU TOÀN CỤC (Hungarian —
    scipy.optimize.linear_sum_assignment), xét TẤT CẢ khả năng ghép cùng
    lúc để chọn phương án có TỔNG khoảng cách nhỏ nhất — không phụ thuộc
    thứ tự, giảm hẳn tình trạng hoán đổi ID.
    """
    max_dist = frame_w * self.max_distance_ratio
    assigned = {}

    track_ids = list(self.tracks.keys())
    n_det, n_tr = len(detections), len(track_ids)

    if n_det > 0 and n_tr > 0:
      # Ma trận chi phí (khoảng cách centroid) — cặp nào vượt quá max_dist
      # bị gán chi phí RẤT LỚN để thuật toán tự loại, không bao giờ ghép.
      cost = np.full((n_det, n_tr), max_dist * 10.0 + 1.0, dtype=np.float64)
      for i, det in enumerate(detections):
        for j, tid in enumerate(track_ids):
          tr = self.tracks[tid]
          d = math.hypot(det["cx"] - tr["cx"], det["cy"] - tr["cy"])
          if d <= max_dist:
            cost[i, j] = d

      from scipy.optimize import linear_sum_assignment
      row_idx, col_idx = linear_sum_assignment(cost)

      matched_det = set()
      matched_tr = set()
      for i, j in zip(row_idx, col_idx):
        if cost[i, j] <= max_dist:
          tid = track_ids[j]
          assigned[tid] = detections[i]
          matched_det.add(i)
          matched_tr.add(tid)
          self.tracks[tid] = {"cx": detections[i]["cx"], "cy": detections[i]["cy"], "lost": 0}

      # Detection không khớp track nào (quá xa mọi track hiện có, hoặc
      # nhiều detection hơn track) -> cấp ID MỚI.
      for i, det in enumerate(detections):
        if i not in matched_det:
          new_id = self.next_id
          self.next_id += 1
          assigned[new_id] = det
          self.tracks[new_id] = {"cx": det["cx"], "cy": det["cy"], "lost": 0}
    else:
      # Không có track cũ nào (VD khung hình đầu tiên) -> mọi detection
      # đều là người MỚI, cấp ID tuần tự như cũ.
      for det in detections:
        new_id = self.next_id
        self.next_id += 1
        assigned[new_id] = det
        self.tracks[new_id] = {"cx": det["cx"], "cy": det["cy"], "lost": 0}

    for tid in list(self.tracks.keys()):
      if tid not in assigned:
        self.tracks[tid]["lost"] += 1
        if self.tracks[tid]["lost"] > self.timeout_frames:
          del self.tracks[tid]

    return assigned


# ==============================================================================
#  3. ĐẶC TRƯNG CHO TRANSFORMER (đã bỏ cảm xúc/thiết bị/3DCNN)
# ==============================================================================


def extract_features(
    face_ok, yaw, pitch, roll, ear_l, ear_r, gaze_ratio, cfg=Config
):
  """Vector đặc trưng cho Transformer — CHỈ dùng tín hiệu khuôn mặt/tư thế

  (Face Mesh), đúng theo pipeline của test_cam.py. Vector được pad về
  đúng Config.FEATURE_DIM để tương thích với model Transformer đã có sẵn.
  """
  base = np.array(
      [
          float(face_ok),
          np.clip(yaw / 90.0, -1, 1),
          np.clip(pitch / 45.0, -1, 1),
          np.clip(roll / 45.0, -1, 1),
          np.clip(ear_l / 0.4, 0, 1),
          np.clip(ear_r / 0.4, 0, 1),
          np.clip(gaze_ratio, 0, 1),
      ],
      dtype=np.float32,
  )
  dim = getattr(cfg, "FEATURE_DIM", 22)
  if len(base) < dim:
    base = np.pad(base, (0, dim - len(base)))
  return base[:dim]


def rule_based_classify(yaw, pitch, ear_l, ear_r, gaze, cnn3d_motion_score=0.0,
                         emotion_stress=False, emotion_label="", face_visible=True):
  """Rule-based fallback khi không có model Transformer — HOẶC dùng làm

  lớp kiểm tra bổ trợ song song với Transformer (xem giamsatthi.py).

  face_visible=False (MỚI — HỢP NHẤT từ nhánh "người không thấy mặt" mà
  trước đây giamsatthi.py tự viết logic riêng, trùng lặp với hàm này):
  khi Face Mesh không dò được mặt (chỉ có bbox thân người từ Faster
  R-CNN), KHÔNG có Yaw/Pitch/EAR/Gaze thật — các tín hiệu đó bị bỏ qua
  hoàn toàn, và cnn3d_motion_score trở thành tín hiệu DUY NHẤT. Vì vậy
  dùng thang điểm RIÊNG (CNN3D_MOTION_SCORE_WEIGHT_NO_FACE, mặc định 8 —
  đủ tự nó vượt ngưỡng CHEATING=5) thay vì trọng số bổ trợ thấp (+3) như
  khi có đủ tín hiệu khuôn mặt.

  cnn3d_motion_score (mặc định 0.0 — tương thích ngược khi không dùng
  TSN/3DCNN): tín hiệu THÔ từ nhánh 3DCNN, xấp xỉ mức độ "chuyển động bất
  thường" — khoảng cách L2 giữa đặc trưng của cửa sổ hiện tại và cửa sổ
  liền trước. Đây là suy luận thô, trọng số CỐ Ý thấp hơn các tín hiệu có
  ý nghĩa vật lý rõ ràng (yaw/pitch/EAR/gaze đo trực tiếp).

  emotion_stress (mặc định False — tương thích ngược khi không dùng nhánh
  cảm xúc): cờ đã được TÍNH SẴN ở nơi gọi (giamsatthi.py) — True khi cảm
  xúc "lo lắng/căng thẳng" (DeepFace) đã kéo dài liên tục đủ lâu
  (EMOTION_STRESS_SUSTAIN_SECONDS). Cũng chỉ là tín hiệu BỔ TRỢ, trọng số
  thấp — cảm xúc lo lắng có thể do nhiều nguyên nhân khác ngoài gian lận
  (áp lực thi cử thông thường), không dùng làm căn cứ duy nhất.
  """
  violations, score = [], 0

  # ── Người KHÔNG THẤY MẶT: chỉ chuyển động toàn thân (TSN) là tín hiệu
  # thật, mọi tín hiệu hình học khuôn mặt (yaw/pitch/EAR/gaze) đều là giá
  # trị GIẢ do nơi gọi truyền vào cho tương thích chữ ký (0.0/0.3/0.5) —
  # KHÔNG được tính điểm, tránh cộng điểm "ảo" từ dữ liệu không có thật.
  if not face_visible:
    if cnn3d_motion_score > Config.CNN3D_MOTION_THRESHOLD:
      score = getattr(Config, "CNN3D_MOTION_SCORE_WEIGHT_NO_FACE", 8)
      violations.append(f"CHUYEN DONG BAT THUONG (3DCNN, {cnn3d_motion_score:.2f}) - KHONG THAY MAT")
    return ("CHEATING" if score >= 5 else "NORMAL"), score, violations

  yaw_valid = abs(yaw) <= 999
  pitch_valid = abs(pitch) <= 999

  if yaw_valid and abs(yaw) > Config.YAW_THRESHOLD:
    score += 5 if abs(yaw) > 35 else 3
    violations.append(f"Quay dau {'TRAI' if yaw < 0 else 'PHAI'} ({yaw:.1f})")

  if pitch_valid and pitch < Config.PITCH_THRESHOLD:
    score += 5 if pitch < -25 else 3
    violations.append(f"Cui xuong ({pitch:.1f})")

  avg_ear = (ear_l + ear_r) / 2
  if avg_ear < Config.EAR_THRESHOLD:
    score += 2
    violations.append(f"Liec mat (EAR={avg_ear:.2f})")

  if gaze < 0.25:
    score += 2
    violations.append("Nhin sang TRAI")
  elif gaze > 0.75:
    score += 2
    violations.append("Nhin sang PHAI")

  # Tín hiệu chuyển động thô từ 3DCNN — trọng số CỐ Ý thấp hơn (+3) vì đây
  # là suy luận xấp xỉ, kém tin cậy hơn nhiều so với góc đo trực tiếp.
  if cnn3d_motion_score > Config.CNN3D_MOTION_THRESHOLD:
    score += 3
    violations.append(
        f"CHUYEN DONG BAT THUONG (3DCNN, {cnn3d_motion_score:.2f})"
    )

  if (
      yaw_valid
      and abs(yaw) > Config.YAW_THRESHOLD
      and cnn3d_motion_score > Config.CNN3D_MOTION_THRESHOLD
  ):
    score += 2
    violations.append("QUAY DAU + CHUYEN DONG BAT THUONG!")

  # Tín hiệu cảm xúc lo lắng/căng thẳng kéo dài — SỬA: KHÔNG còn cộng điểm
  # vào score nữa (trước đây +2). Cảm xúc lo lắng/căng thẳng khi thi có
  # RẤT NHIỀU nguyên nhân hoàn toàn bình thường (áp lực thi cử, tính cách
  # hay hồi hộp, mất ngủ...) — không có mối liên hệ nhân quả đáng tin cậy
  # với hành vi gian lận để dùng làm CĂN CỨ TÍNH ĐIỂM, dù trọng số thấp.
  # Vẫn GHI NHẬN vào violations để hiển thị/log tham khảo cho giám thị,
  # nhưng gắn rõ tiền tố "THAM KHAO" để không bị hiểu nhầm là 1 vi phạm.
  if emotion_stress:
    violations.append(f"THAM KHAO - CAM XUC LO LANG KEO DAI ({emotion_label})")

  return ("CHEATING" if score >= 5 else "NORMAL"), score, violations


def debounce_label(prev_label, cheat_streak, normal_streak, raw_label,
                    enter_frames=3, exit_frames=5):
  """Chống nhiễu 1 khung hình (debounce/hysteresis) cho 1 nhãn CHEATING/

  NORMAL thô, tính lại từ đầu mỗi khung hình (VD rule_based_classify()) —
  vốn dễ "nháy" qua lại chỉ vì tín hiệu đo đạc rung nhẹ quanh ngưỡng.

  Cách hoạt động: chỉ THỰC SỰ đổi trạng thái khi raw_label đã LIÊN TỤC
  giữ nguyên đủ enter_frames (khi đang NORMAL muốn chuyển sang CHEATING)
  hoặc đủ exit_frames (khi đang CHEATING muốn thoát ra) — "vào nhanh, ra
  chậm hơn 1 chút" giúp không bỏ sót hành vi thật nhưng vẫn lọc được
  nhiễu 1-2 khung hình đơn lẻ.

  Trả về (label_đã_debounce, cheat_streak_mới, normal_streak_mới) — nơi
  gọi tự lưu lại 3 giá trị này (VD trong PersonState) để truyền vào lần
  gọi kế tiếp.
  """
  if raw_label == "CHEATING":
    cheat_streak += 1
    normal_streak = 0
  else:
    normal_streak += 1
    cheat_streak = 0

  if prev_label != "CHEATING" and cheat_streak >= enter_frames:
    return "CHEATING", cheat_streak, normal_streak
  if prev_label == "CHEATING" and normal_streak >= exit_frames:
    return "NORMAL", cheat_streak, normal_streak
  return prev_label, cheat_streak, normal_streak


# ==============================================================================
#  4. BỘ PHÂN LOẠI HÀNH VI — TRANSFORMER ENCODER (Self-Attention)
# ==============================================================================

if TORCH_AVAILABLE:

  class PositionalEncoding(nn.Module):
    """Cộng thêm thông tin vị trí thời gian vào từng bước (Transformer

    không có khái niệm thứ tự tự nhiên như RNN).
    """

    def __init__(self, d_model, max_len=200):
      super().__init__()
      pe = torch.zeros(max_len, d_model)
      position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
      div_term = torch.exp(
          torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
      )
      pe[:, 0::2] = torch.sin(position * div_term)
      pe[:, 1::2] = torch.cos(position * div_term)
      self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
      return x + self.pe[:, : x.size(1), :]

  class CheatingTransformer(nn.Module):
    """Transformer Encoder phân loại hành vi gian lận từ chuỗi đặc trưng.

    Input : (batch, seq_len, feature_dim)
    Output: (batch, 2) — [normal, cheating]
    """

    def __init__(
        self,
        input_dim=22,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        num_classes=2,
        dropout=0.3,
    ):
      super().__init__()
      self.input_proj = nn.Linear(input_dim, d_model)
      self.pos_encoding = PositionalEncoding(d_model)
      encoder_layer = nn.TransformerEncoderLayer(
          d_model=d_model,
          nhead=nhead,
          dim_feedforward=dim_feedforward,
          dropout=dropout,
          batch_first=True,
          activation="relu",
      )
      self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
      self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
      self.classifier = nn.Sequential(
          nn.Linear(d_model, 64),
          nn.ReLU(),
          nn.Dropout(dropout),
          nn.Linear(64, num_classes),
      )

    def forward(self, x):
      b = x.size(0)
      x = self.input_proj(x)
      cls = self.cls_token.expand(b, -1, -1)
      x = torch.cat([cls, x], dim=1)
      x = self.pos_encoding(x)
      out = self.encoder(x)
      cls_out = out[:, 0, :]
      return self.classifier(cls_out)


def load_transformer_model(path=None):
  """Tải model Transformer đã huấn luyện, trả về None nếu chưa có/PyTorch chưa cài."""
  path = path or Config.MODEL_PATH
  if not TORCH_AVAILABLE or not os.path.exists(path):
    return None
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = CheatingTransformer(input_dim=Config.FEATURE_DIM).to(device)
  try:
    state = torch.load(path, map_location=device)
    state_dict = (
        state["state_dict"]
        if isinstance(state, dict) and "state_dict" in state
        else state
    )
    model.load_state_dict(state_dict)
  except Exception as e:
    print(f"[WARN] Không nạp được Transformer ({path}): {str(e)[:160]}\n"
          f"       Thường do model cũ khác số chiều đặc trưng (hiện là {Config.FEATURE_DIM}). "
          f"Hệ thống chạy bằng luật hành vi cho tới khi huấn luyện lại (train_transformer.py).")
    return None
  model.eval()
  print(f"[OK] Transformer loaded: {path} ({device})")
  return model


def classify_with_transformer(model, feat_buf, device=None):
  """Chạy suy luận Transformer trên buffer đặc trưng (deque đã đủ SEQ_LEN phần tử).

  Trả về (label, raw_prob) với raw_prob = xác suất CHEATING trong [0,1].
  """
  device = device or next(model.parameters()).device
  seq_t = (
      torch.tensor(np.stack(list(feat_buf)), dtype=torch.float32)
      .unsqueeze(0)
      .to(device)
  )
  with torch.no_grad():
    out = model(seq_t)
    if out.ndim > 1 and out.shape[-1] > 1:
      raw_prob = torch.softmax(out, dim=-1)[0, 1].item()
    else:
      raw_prob = torch.sigmoid(out).item()
  label = "CHEATING" if raw_prob >= Config.CHEAT_THRESHOLD else "NORMAL"
  return label, raw_prob


def train_transformer_model(
    X_train, y_train, X_val, y_val, epochs=50, batch_size=32, lr=1e-3
):
  """Train Transformer model từ dataset.

  X shape: (N, SEQ_LEN, FEATURE_DIM), y: (N,) với 0=Normal, 1=Cheating.
  """
  if not TORCH_AVAILABLE:
    print("[ERROR] Cần cài PyTorch")
    return
  from torch.utils.data import DataLoader, TensorDataset

  os.makedirs("models", exist_ok=True)
  Xtr = torch.tensor(X_train, dtype=torch.float32)
  ytr = torch.tensor(y_train, dtype=torch.long)
  Xv = torch.tensor(X_val, dtype=torch.float32)
  yv = torch.tensor(y_val, dtype=torch.long)
  tl = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
  vl = DataLoader(TensorDataset(Xv, yv), batch_size=batch_size)
  model = CheatingTransformer(input_dim=Config.FEATURE_DIM)
  opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
  sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)
  crit = nn.CrossEntropyLoss()
  best = 0.0
  for ep in range(1, epochs + 1):
    model.train()
    tl_sum = 0
    for xb, yb in tl:
      opt.zero_grad()
      loss = crit(model(xb), yb)
      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      opt.step()
      tl_sum += loss.item()
    model.eval()
    corr = tot = vl_sum = 0
    with torch.no_grad():
      for xb, yb in vl:
        out = model(xb)
        vl_sum += crit(out, yb).item()
        corr += (out.argmax(1) == yb).sum().item()
        tot += yb.size(0)
    acc = corr / tot * 100
    sched.step(vl_sum)
    print(
        f"Epoch {ep:03d}/{epochs} | Loss:{tl_sum/len(tl):.4f} | Val:{acc:.1f}%"
    )
    if acc > best:
      best = acc
      torch.save(model.state_dict(), "models/transformer_model.pt")
      print(f"  → Saved (val={acc:.1f}%)")
  print(f"\n[DONE] Best val: {best:.1f}% | models/transformer_model.pt")


# ==============================================================================
#  4b. NHÁNH HÀNH VI/TƯ THẾ — TSN (mmaction2), CHẠY ĐỘC LẬP VỚI TRANSFORMER
# ==============================================================================


def init_3dcnn_model(cfg=Config):
  """Tải model TSN đã tự huấn luyện (mmaction2).

  Trả về (model, transform) — transform luôn None (TSN tự tiền xử lý trong
  extract_3dcnn_clip_features()), giữ cùng chữ ký hàm cho gọn.
  """
  # Nếu TSN bị tắt trong Config, thoát êm ngay lập tức không in cảnh báo
  if not getattr(cfg, "TSN_ENABLED", True):
    return None, None

  if not (TORCH_AVAILABLE and MMACTION_AVAILABLE):
    print(
        "[WARN] Thiếu PyTorch/mmaction2 — nhánh hành vi (TSN) sẽ KHÔNG chạy "
        "(cnn3d_motion_score sẽ luôn là 0)."
    )
    return None, None
  if not (
      os.path.exists(cfg.MMACTION_CONFIG)
      and os.path.exists(cfg.MMACTION_CHECKPOINT)
  ):
    print(
        f"[WARN] Không tìm thấy config/checkpoint TSN:\n"
        f"       Config    : {cfg.MMACTION_CONFIG}\n"
        f"       Checkpoint: {cfg.MMACTION_CHECKPOINT}\n"
        f"       -> nhánh hành vi (TSN) sẽ KHÔNG chạy."
    )
    return None, None
  try:
    import functools

    _original_torch_load = torch.load
    torch.load = functools.partial(_original_torch_load, weights_only=False)
    try:
      recognizer = init_recognizer(
          cfg.MMACTION_CONFIG,
          cfg.MMACTION_CHECKPOINT,
          device=cfg.MMACTION_DEVICE,
      )
    finally:
      torch.load = _original_torch_load
    recognizer.eval()

    project = nn.Linear(
        cfg.MMACTION_BACKBONE_FEATURE_DIM, cfg.CNN3D_FEATURE_DIM
    )
    project.eval()

    print(
        f"[OK] mmaction2 TSN loaded (config={os.path.basename(cfg.MMACTION_CONFIG)}, "
        f"checkpoint={os.path.basename(cfg.MMACTION_CHECKPOINT)}). "
        f"CNN3D_FEATURE_DIM={cfg.CNN3D_FEATURE_DIM} num_segments={cfg.TSN_NUM_SEGMENTS} "
        f"buffer_len={cfg.CNN3D_CLIP_LEN}"
    )
    return {"recognizer": recognizer, "project": project}, None
  except Exception as e:
    print(f"[WARN] Không tải được model TSN: {e!r}")
    print(
        "[WARN] Kiểm tra lại: (1) MMACTION_CONFIG/MMACTION_CHECKPOINT có trỏ đúng "
        "file thật không; (2) đã cài đủ mmengine+mmcv+mmaction2 chưa (mim list). "
        "Hệ thống vẫn chạy được (Transformer/rule-based chính không bị ảnh hưởng), "
        "chỉ thiếu tín hiệu hành vi bổ trợ."
    )
    return None, None


def extract_3dcnn_clip_features(model, transform, frame_clip, cfg=Config):
  """Chạy TSN trên các khung hình lấy mẫu rải đều từ frame_clip (buffer

  CNN3D_CLIP_LEN khung hình — vùng ảnh 1 người, không phải cả khung hình
  camera) → trả về vector numpy CNN3D_FEATURE_DIM chiều. Trả về vector 0
  nếu model chưa sẵn sàng hoặc buffer chưa đủ khung hình.
  """
  if model is None or len(frame_clip) < cfg.CNN3D_CLIP_LEN:
    return np.zeros(cfg.CNN3D_FEATURE_DIM, dtype=np.float32)
  try:
    recognizer = model["recognizer"]
    project = model["project"]

    n_seg = cfg.TSN_NUM_SEGMENTS
    idx = np.linspace(0, len(frame_clip) - 1, n_seg).astype(int)
    sampled = [frame_clip[i] for i in idx]

    short_side = cfg.CNN3D_RESIZE_SHORT_SIDE
    crop = cfg.CNN3D_PROC_SIZE
    processed = []
    for f in sampled:
      f_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
      h, w = f_rgb.shape[:2]
      scale = short_side / min(h, w)
      f_resized = cv2.resize(
          f_rgb,
          (
              max(crop, int(round(w * scale))),
              max(crop, int(round(h * scale))),
          ),
      )
      H, W = f_resized.shape[:2]
      top, left = (H - crop) // 2, (W - crop) // 2
      processed.append(f_resized[top : top + crop, left : left + crop])

    clip_arr = np.stack(processed, axis=0).astype(
        np.float32
    )  # (n_seg, crop, crop, 3)

    mean = np.array(cfg.MMACTION_NORM_MEAN, dtype=np.float32)
    std = np.array(cfg.MMACTION_NORM_STD, dtype=np.float32)
    clip_arr = (clip_arr - mean) / std

    imgs = torch.from_numpy(clip_arr).permute(0, 3, 1, 2).contiguous().float()
    imgs = imgs.unsqueeze(0)  # (T, C, H, W) → (1, T, C, H, W)

    with torch.no_grad():
      feat = recognizer.extract_feat(imgs)
      if isinstance(feat, (tuple, list)):
        feat = feat[0]
      if feat.dim() > 2:
        feat = feat.mean(dim=list(range(2, feat.dim())))
      feat = feat.mean(dim=0, keepdim=True)  # temporal consensus
      out = torch.tanh(project(feat))  # giảm chiều

    return out.squeeze(0).cpu().numpy().astype(np.float32)
  except Exception as e:
    print(f"[TSN][WARN] Lỗi khi trích đặc trưng: {e!r}")
    return np.zeros(cfg.CNN3D_FEATURE_DIM, dtype=np.float32)


def crop_person_region(frame, bbox, buffer_width=None, margin_ratio=0.6):
  """Cắt vùng ảnh quanh 1 người từ bbox khuôn mặt (do process_faces_frame trả

  về), MỞ RỘNG thêm margin_ratio để bao trọn phần thân trên/tay (TSN cần
  thấy chuyển động cơ thể, không chỉ khuôn mặt) — rồi resize về chiều rộng
  CNN3D_BUFFER_WIDTH.
  """
  h, w = frame.shape[:2]
  x0, y0, x1, y1 = bbox
  bw, bh = x1 - x0, y1 - y0
  mx, my = int(bw * margin_ratio), int(bh * margin_ratio * 2.2)
  cx0, cy0 = max(0, x0 - mx), max(0, y0 - int(my * 0.3))
  cx1, cy1 = min(w, x1 + mx), min(h, y1 + my)
  crop = frame[cy0:cy1, cx0:cx1]
  if crop.size == 0:
    return None
  buffer_width = buffer_width or Config.CNN3D_BUFFER_WIDTH
  ch, cw = crop.shape[:2]
  if cw > buffer_width:
    scale = buffer_width / cw
    crop = cv2.resize(crop, (buffer_width, max(1, int(ch * scale))))
  return crop


def crop_face_region(frame, bbox, margin_ratio=None):
  """Cắt CHỈ vùng khuôn mặt (không lấy thân trên như crop_person_region)."""
  h, w = frame.shape[:2]
  x0, y0, x1, y1 = bbox
  margin_ratio = (
      margin_ratio
      if margin_ratio is not None
      else getattr(Config, "FACE_MARGIN_RATIO", 0.1)
  )
  mx = int((x1 - x0) * margin_ratio)
  my = int((y1 - y0) * margin_ratio)
  cx0, cy0 = max(0, x0 - mx), max(0, y0 - my)
  cx1, cy1 = min(w, x1 + mx), min(h, y1 + my)
  crop = frame[cy0:cy1, cx0:cx1]
  return crop if crop.size > 0 else None


# ==============================================================================
#  5. QUY ĐỔI RA % GIAN LẬN
# ==============================================================================

CONFIDENCE_LEVEL_COLORS = {
    "RẤT CAO": (0, 0, 255),
    "CAO": (0, 80, 255),
    "TRUNG BÌNH": (0, 165, 255),
    "THẤP": (0, 200, 80),
}


def confidence_level_color(confidence_level: str, default=(200, 200, 200)):
  return CONFIDENCE_LEVEL_COLORS.get(confidence_level, default)


def compute_fraud_confidence(score, violations, sustained_secs):
  """Trả về (fraud_pct, assessment, confidence_level) từ điểm rule-based

  (hoặc xác suất Transformer * 10, xem giamsatthi.py) + thời gian bền vững.
  """
  base = min(score / 10.0, 1.0) * 65.0
  time_bonus = min(sustained_secs / 8.0, 1.0) * 20.0
  n_viol = len(violations)
  combo_bonus = min(n_viol * 3.0, 15.0)
  pct = min(base + time_bonus + combo_bonus, 100.0)

  if pct >= 85:
    lvl, msg = "RẤT CAO", "Gần chắc chắn gian lận - can thiep ngay"
  elif pct >= 65:
    lvl, msg = "CAO", "Rat nghi ngo gian lan - quan sat them"
  elif pct >= 40:
    lvl, msg = "TRUNG BÌNH", "Co dau hieu dang ngo - theo doi tiep"
  else:
    lvl, msg = "THẤP", "Dau hieu nhe - tiep tuc theo doi"

  return round(pct, 1), msg, lvl


# ==============================================================================
#  6. HUD — GIAO DIỆN HIỂN THỊ (ĐA NGƯỜI, TINH GỌN)
# ==============================================================================


def draw_multi_hud(frame, person_states, fps):
  """HUD giám sát NHIỀU NGƯỜI cùng lúc:

    - 1 thanh tổng quan góc trên-trái: tổng số người đang giám sát + số
      người đang bị nghi ngờ gian lận + FPS.
    - Mỗi người 1 khung bao quanh mặt (màu xanh=NORMAL, đỏ/cam=CHEATING)
      kèm nhãn nhỏ phía trên: "#ID  TRẠNG THÁI  xx%".

  person_states: dict {track_id: {"bbox", "label", "fraud_pct", "color"}}
  """
  f = Config.FONT
  n_total = len(person_states)
  # SỬA LỖI NGHIÊM TRỌNG (bug thật, đã xác nhận qua thực tế sử dụng): TRƯỚC
  # ĐÂY so sánh p.get("label") == "CHEATING" — nhưng "label" luôn là CHUỖI
  # HIỂN THỊ ĐẦY ĐỦ (VD "[D1-G2] CHEATING [CAO]"), KHÔNG BAO GIỜ
  # đúng bằng chính xác chuỗi "CHEATING" -> n_cheat LUÔN LUÔN = 0, dòng
  # "Nghi ngo: 0" không bao giờ cập nhật dù có người đang bị nghi ngờ thật.
  # Giờ dùng cờ boolean "is_cheating" riêng (nơi gọi trong giamsatthi.py tự
  # gán đúng True/False khi tạo person_states), không phụ thuộc vào việc
  # so khớp chuỗi hiển thị (vốn có thể đổi định dạng bất cứ lúc nào).
  n_cheat = sum(
      1 for p in person_states.values() if p.get("is_cheating")
  )
  bar_color = (0, 0, 255) if n_cheat > 0 else (0, 200, 80)

  cv2.rectangle(frame, (20, 20), (430, 72), (25, 25, 25), -1)
  cv2.rectangle(frame, (20, 20), (430, 72), bar_color, 2)
  cv2.putText(
      frame,
      f"Giam sat: {n_total} nguoi | Nghi ngo: {n_cheat}",
      (32, 48),
      f,
      0.6,
      bar_color,
      2,
  )
  cv2.putText(
      frame, f"FPS:{fps:4.1f}", (32, 66), f, 0.45, (180, 180, 180), 1
  )

  fh, fw = frame.shape[:2]
  for tid, st in person_states.items():
    x0, y0, x1, y1 = st["bbox"]
    color = st["color"]

    label = st["label"]
    (tw, th), _ = cv2.getTextSize(label, f, 0.5, 2)
    # Không vẽ khung bao quanh cả người nữa (chiếm nhiều diện tích) — chỉ vẽ nhãn nhỏ gọn
    # + 1 mũi tên trỏ thẳng từ nhãn vào người, đỡ che khuất bài làm/khuôn mặt.
    tip_x = max(0, min(fw - 1, (x0 + x1) // 2))
    tip_y = max(0, min(fh - 1, y0 + int(0.15 * max(1, y1 - y0))))  # điểm đích: gần đỉnh đầu trong khung người
    label_x = max(0, min(fw - tw - 10, tip_x - (tw + 10) // 2))    # nhãn canh giữa quanh điểm đích, không tràn khung hình
    ly0 = max(0, y0 - th - 10)
    cv2.rectangle(frame, (label_x, ly0), (label_x + tw + 10, ly0 + th + 8), color, -1)
    text_color = (0, 0, 0) if color == (0, 200, 80) else (255, 255, 255)
    cv2.putText(frame, label, (label_x + 5, ly0 + th + 2), f, 0.5, text_color, 2)

    arrow_from = (label_x + (tw + 10) // 2, ly0 + th + 8)
    if math.hypot(arrow_from[0] - tip_x, arrow_from[1] - tip_y) >= 6:  # nhãn với người quá gần thì khỏi vẽ mũi tên (tránh vẽ chấm)
      cv2.arrowedLine(frame, arrow_from, (tip_x, tip_y), color, 2, tipLength=0.25)
  return frame


# ==============================================================================
#  7. ẢNH BẰNG CHỨNG
# ==============================================================================


def capture_evidence_image(
    frame,
    fraud_pct,
    assessment,
    confidence_level,
    violations,
    person_id=None,
    bbox=None,
):
  """Tạo 1 ảnh bằng chứng: toàn bộ khung hình + overlay thông tin."""
  ev = frame.copy()
  h, w = ev.shape[:2]
  f = Config.FONT
  col = confidence_level_color(confidence_level)

  if bbox is not None:
    x0, y0, x1, y1 = bbox
    cv2.rectangle(ev, (x0, y0), (x1, y1), (0, 0, 255), 3)
    cv2.putText(
        ev, f"NGUOI #{person_id}", (x0, max(20, y0 - 10)), f, 0.6, (0, 0, 255), 2
    )

  ov = ev.copy()
  cv2.rectangle(ov, (0, 0), (w, 130), (10, 10, 10), -1)
  cv2.addWeighted(ov, 0.7, ev, 0.3, 0, ev)

  title = (
      "=== BANG CHUNG GIAN LAN ==="
      if person_id is None
      else f"=== BANG CHUNG GIAN LAN - NGUOI #{person_id} ==="
  )
  cv2.putText(ev, title, (w // 2 - 260, 28), f, 0.75, (255, 255, 255), 2)
  cv2.putText(
      ev, f"Do tin cay gian lan: {fraud_pct:.1f}%", (10, 60), f, 0.75, col, 2
  )

  bx, by, bw2, bh2 = 10, 72, min(w - 20, 600), 14
  cv2.rectangle(ev, (bx, by), (bx + bw2, by + bh2), (60, 60, 60), -1)
  cv2.rectangle(
      ev, (bx, by), (bx + int(fraud_pct / 100 * bw2), by + bh2), col, -1
  )
  cv2.rectangle(ev, (bx, by), (bx + bw2, by + bh2), (180, 180, 180), 1)

  cv2.putText(ev, f"Muc do: {confidence_level}", (10, 105), f, 0.65, col, 2)
  cv2.putText(ev, assessment, (10, 128), f, 0.5, (200, 200, 200), 1)

  if violations:
    ov3 = ev.copy()
    cv2.rectangle(ov3, (0, h - 75), (520, h), (0, 0, 80), -1)
    cv2.addWeighted(ov3, 0.55, ev, 0.45, 0, ev)
    for i, v in enumerate(violations[:4]):
      cv2.putText(
          ev, f"  * {v}", (10, h - 66 + i * 24), f, 0.5, (80, 180, 255), 1
      )

  cv2.putText(
      ev,
      time.strftime("%Y-%m-%d %H:%M:%S"),
      (w - 250, h - 10),
      f,
      0.5,
      (160, 160, 160),
      1,
  )
  return ev


# ==============================================================================
#  8. GHI LOG CSV + BÁO CÁO JSON
# ==============================================================================


class CheatingLogger:

  def __init__(self, log_file, evidence_dir):
    self.evidence_dir = evidence_dir
    os.makedirs(evidence_dir, exist_ok=True)
    self.log_file = open(log_file, "w", newline="", encoding="utf-8")
    self.writer = csv.writer(self.log_file)
    self.writer.writerow([
        "timestamp",
        "frame_id",
        "person_id",
        "seat_label",
        "yaw",
        "pitch",
        "ear_left",
        "ear_right",
        "gaze_ratio",
        "score",
        "label",
        "violations",
        "screenshot_path",
        "fraud_pct",
        "confidence_level",
        "sustained_seconds",
        "emotion",
    ])
    self.frame_count = 0
    # SỬA: trước đây gọi self.log_file.flush() SAU MỖI DÒNG (mỗi khung
    # hình, ~30 lần/giây) — flush() ép hệ điều hành ghi xuống đĩa NGAY
    # LẬP TỨC, là 1 syscall tương đối tốn kém khi gọi liên tục với tần
    # suất cao, tích lũy lại gây tải I/O không cần thiết trong 1 phiên thi
    # dài. Giờ CHỈ flush định kỳ (mỗi FLUSH_EVERY_N_ROWS dòng, mặc định
    # ~1 giây ở 30fps) — vẫn ghi ĐẦY ĐỦ MỌI khung hình vào buffer của
    # Python/hệ điều hành như cũ (KHÔNG mất dữ liệu, chỉ trì hoãn việc ép
    # xuống đĩa), và luôn flush() lần cuối trong close() để đảm bảo không
    # mất dữ liệu khi dừng phiên.
    self.FLUSH_EVERY_N_ROWS = 30

  def log(
      self,
      yaw,
      pitch,
      ear_l,
      ear_r,
      gaze,
      score,
      label,
      violations,
      fraud_pct=0.0,
      confidence_level="",
      sustained_secs=0.0,
      screenshot_path="",
      person_id=None,
      emotion="",
      seat_label="",
  ):
    self.frame_count += 1
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    self.writer.writerow([
        ts,
        self.frame_count,
        person_id if person_id is not None else "",
        seat_label,
        f"{yaw:.2f}",
        f"{pitch:.2f}",
        f"{ear_l:.3f}",
        f"{ear_r:.3f}",
        f"{gaze:.3f}",
        score,
        label,
        " | ".join(violations),
        screenshot_path,
        f"{fraud_pct:.1f}",
        confidence_level,
        f"{sustained_secs:.1f}",
        emotion,
    ])
    if self.frame_count % self.FLUSH_EVERY_N_ROWS == 0:
      self.log_file.flush()

  def close(self):
    self.log_file.flush()
    self.log_file.close()


class SessionReporter:

  def __init__(self, report_file, report_dir, exam_room, proctor):
    self.report_file = report_file
    self.report_dir = report_dir
    self.exam_room = exam_room
    self.proctor = proctor or self._username()
    os.makedirs(report_dir, exist_ok=True)
    self.session_id = str(uuid.uuid4())[:12]
    self.date_str = datetime.now().strftime("%Y-%m-%d")
    self.start_time = datetime.now()
    self.start_ts = time.time()
    # SỬA: trước đây _fy/_fp/_fel/_fer/_fg/_fsc là LIST, phình to VÔ HẠN
    # suốt phiên thi (mẫu mỗi 5 khung hình — phiên thi 3 tiếng ở 30fps vẫn
    # là ~65,000 phần tử MỖI mảng, x6 mảng). Nhưng cuối cùng chỉ dùng để
    # tính TRUNG BÌNH (self._avg trong _build()) — không cần giữ lại từng
    # mẫu. Đổi sang thống kê chạy (running sum + count, thuật toán O(1) bộ
    # nhớ bất kể phiên dài bao lâu) — kết quả trung bình giống hệt, chỉ
    # khác cách tính, không đổi giao diện/báo cáo đầu ra.
    self._stat_sum = {"yaw": 0.0, "pitch": 0.0, "ear_l": 0.0, "ear_r": 0.0, "gaze": 0.0, "score": 0.0}
    self._stat_count = 0
    self._total = 0
    self._cheat = 0
    self.incidents = []
    self._inc_counter = 0
    self._active_incidents = {}
    print(f"[SESSION] ID:{self.session_id} | {self.date_str} | {self.exam_room}")

  def _username(self):
    try:
      return getpass.getuser()
    except Exception:
      return "unknown"

  def _sysinfo(self):
    return {
        "os": platform.system(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "user": self.proctor,
    }

  def _cfg_snap(self):
    c = Config
    return {
        "camera_id": c.CAMERA_ID,
        "yaw_threshold": c.YAW_THRESHOLD,
        "pitch_threshold": c.PITCH_THRESHOLD,
        "ear_threshold": c.EAR_THRESHOLD,
        "cheat_threshold": c.CHEAT_THRESHOLD,
    }

  def update_frame(self, yaw, pitch, ear_l, ear_r, gaze, score, final_label):
    self._total += 1
    if final_label == "CHEATING":
      self._cheat += 1
    if self._total % 5 == 0:
      self._stat_sum["yaw"] += yaw
      self._stat_sum["pitch"] += pitch
      self._stat_sum["ear_l"] += ear_l
      self._stat_sum["ear_r"] += ear_r
      self._stat_sum["gaze"] += gaze
      self._stat_sum["score"] += score
      self._stat_count += 1

  def has_active_incident(self, person_id):
    return person_id in self._active_incidents

  def open_incident(
      self, person_id, timestamp, violations, yaw, pitch, ear_l, seat_label=""
  ):
    if person_id in self._active_incidents:
      return
    self._inc_counter += 1
    self._active_incidents[person_id] = {
        "incident_id": self._inc_counter,
        "person_id": person_id,
        "seat_label": seat_label,
        "detected_at": timestamp,
        "violations": list(violations),
        "_ts": time.time(),
        "_yw": [round(yaw, 2)],
        "_pt": [round(pitch, 2)],
        "_er": [round(ear_l, 3)],
        "fraud_pct": 0.0,
        "confidence_level": "",
        "assessment": "",
        "screenshot_path": "",
        "verify_status": "unverified",
        "proctor_note": "",
    }

  def update_incident(
      self,
      person_id,
      fraud_pct,
      confidence_level,
      assessment,
      yaw,
      pitch,
      ear_l,
      violations,
  ):
    inc = self._active_incidents.get(person_id)
    if not inc:
      return
    if fraud_pct >= inc["fraud_pct"]:
      inc["fraud_pct"] = round(fraud_pct, 1)
      inc["confidence_level"] = confidence_level
      inc["assessment"] = assessment
    inc["_yw"].append(round(yaw, 2))
    inc["_pt"].append(round(pitch, 2))
    inc["_er"].append(round(ear_l, 3))
    for v in violations:
      if v not in inc["violations"]:
        inc["violations"].append(v)

  def set_incident_screenshot(self, person_id, path):
    inc = self._active_incidents.get(person_id)
    if inc:
      inc["screenshot_path"] = path

  def close_incident(self, person_id):
    inc = self._active_incidents.pop(person_id, None)
    if not inc:
      return
    el = round(time.time() - inc.pop("_ts"), 2)
    yw = inc.pop("_yw", [0.0])
    pt = inc.pop("_pt", [0.0])
    er = inc.pop("_er", [0.3])
    inc["duration_seconds"] = el
    inc["max_yaw"] = round(max(abs(v) for v in yw), 2)
    inc["min_pitch"] = round(min(pt), 2)
    inc["min_ear"] = round(min(er), 3)
    self.incidents.append(inc)
    print(
        f"[INCIDENT #{inc['incident_id']}] Người #{person_id} | {el:.1f}s |"
        f" Fraud:{inc['fraud_pct']}% [{inc['confidence_level']}]"
    )

  def close_all_incidents(self):
    for person_id in list(self._active_incidents.keys()):
      self.close_incident(person_id)

  def _avg(self, lst):
    return round(sum(lst) / len(lst), 3) if lst else 0.0

  def _stat_avg(self, key):
    return round(self._stat_sum[key] / self._stat_count, 3) if self._stat_count else 0.0

  def _build(self):
    cf = [i["fraud_pct"] for i in self.incidents]
    return {
        "session_id": self.session_id,
        "start_time": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": round(time.time() - self.start_ts, 2),
        "exam_room": self.exam_room,
        "proctor": self.proctor,
        "system_info": self._sysinfo(),
        "config_snapshot": self._cfg_snap(),
        "summary": {
            "total_frames": self._total,
            "cheating_frames": self._cheat,
            "cheating_rate_pct": round(
                self._cheat / max(self._total, 1) * 100, 2
            ),
            "total_cheating_incidents": len(self.incidents),
            "max_fraud_pct": round(max(cf, default=0.0), 1),
            "avg_fraud_pct_when_cheating": round(self._avg(cf), 1),
        },
        "frame_stats": {
            "avg_yaw": self._stat_avg("yaw"),
            "avg_pitch": self._stat_avg("pitch"),
            "avg_ear_left": self._stat_avg("ear_l"),
            "avg_ear_right": self._stat_avg("ear_r"),
            "avg_gaze": self._stat_avg("gaze"),
            "avg_rule_score": self._stat_avg("score"),
        },
        "incidents": self.incidents,
    }

  def save(self):
    self.close_all_incidents()
    rec = self._build()

    data = {}
    if os.path.exists(self.report_file):
      try:
        with open(self.report_file, encoding="utf-8") as f:
          data = json.load(f)
      except Exception:
        pass

    de = data.setdefault(
        self.date_str,
        {"date": self.date_str, "total_sessions": 0, "sessions": []},
    )
    idx = de["total_sessions"] + 1
    rec["session_index"] = idx
    de["total_sessions"] = idx
    de["sessions"].append(rec)

    all_inc = [i for s in de["sessions"] for i in s.get("incidents", [])]
    de["day_summary"] = {
        "total_sessions": idx,
        "total_incidents_today": len(all_inc),
        "max_fraud_pct_today": round(
            max((i["fraud_pct"] for i in all_inc), default=0.0), 1
        ),
    }
    with open(self.report_file, "w", encoding="utf-8") as f:
      json.dump(data, f, ensure_ascii=False, indent=2)

    dp = os.path.join(self.report_dir, f"{self.date_str}.json")
    dd = {}
    if os.path.exists(dp):
      try:
        with open(dp, encoding="utf-8") as f:
          dd = json.load(f)
      except Exception:
        pass
    dd.setdefault("date", self.date_str)
    dd.setdefault("total_sessions", 0)
    dd.setdefault("sessions", [])
    dd["total_sessions"] = idx
    dd["day_summary"] = de["day_summary"]
    eids = {s["session_id"] for s in dd["sessions"]}
    if rec["session_id"] not in eids:
      dd["sessions"].append(rec)
    else:
      dd["sessions"] = [
          rec if s["session_id"] == rec["session_id"] else s
          for s in dd["sessions"]
      ]
    with open(dp, "w", encoding="utf-8") as f:
      json.dump(dd, f, ensure_ascii=False, indent=2)

    s = rec["summary"]
    print(f"\n{'='*56}\n  BÁO CÁO PHIÊN #{idx} — {self.date_str}")
    print(f"  Phòng:{self.exam_room} | Giám thị:{self.proctor}")
    print(
        f"  Frame:{s['total_frames']:,} | Gian lận:{s['cheating_frames']:,}"
        f" ({s['cheating_rate_pct']:.2f}%)"
    )
    print(
        f"  Sự kiện:{s['total_cheating_incidents']} | Fraud"
        f" max:{s['max_fraud_pct']}%"
    )
    print(f"  JSON tổng: {self.report_file} | Theo ngày: {dp}\n{'='*56}\n")
    return rec, dp


def safe_folder_name(s):
  s = (s or "unknown").strip()
  s = re.sub(r'[<>:"/\\|?*]', "_", s)
  s = s.replace(" ", "_")
  return s or "unknown"


def get_daily_evidence_dir(base_dir=None, room=None):
  d = os.path.join(
      base_dir or Config.EVIDENCE_DIR, datetime.now().strftime("%Y-%m-%d")
  )
  if room:
    d = os.path.join(d, safe_folder_name(room))
  os.makedirs(d, exist_ok=True)
  return d


def finalize_daily_evidence(
    base_dir, date_str, incidents, session_id, exam_room, proctor, room=None
):
  daily_dir = os.path.join(base_dir, date_str)
  if room:
    daily_dir = os.path.join(daily_dir, safe_folder_name(room))
  if not os.path.isdir(daily_dir):
    return
  imgs = sorted(
      f
      for f in os.listdir(daily_dir)
      if f.lower().endswith((".jpg", ".png", ".jpeg"))
  )
  lines = [
      "=" * 60,
      "  TỔNG HỢP BẰNG CHỨNG GIÁM SÁT THI CỬ",
      f"  Ngày: {date_str} | Phòng: {exam_room} | Giám thị: {proctor}",
      f"  Session ID: {session_id}",
      "=" * 60,
      "",
      f"SỰ KIỆN GIAN LẬN: {len(incidents)}",
      "-" * 60,
  ]
  for i, inc in enumerate(incidents, 1):
    lines.append(
        f"  #{i:02d} {inc.get('detected_at','')} |"
        f" Fraud:{inc.get('fraud_pct',0):.1f}%"
        f" [{inc.get('confidence_level','')}]"
    )
    lines.append(
        f"       {inc.get('duration_seconds',0):.1f}s |"
        f" {', '.join(inc.get('violations',[]))}"
    )
    if inc.get("screenshot_path"):
      lines.append(f"       Ảnh: {os.path.basename(inc['screenshot_path'])}")
    lines.append("")
  lines += ["-" * 60, f"DANH SÁCH ẢNH: {len(imgs)} file", "-" * 60]
  for img in imgs:
    sz = os.path.getsize(os.path.join(daily_dir, img)) // 1024
    lines.append(f"  {img}  ({sz} KB)")
  lines += [
      "",
      "=" * 60,
      f"  Tổng hợp lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
      "=" * 60,
  ]
  sp = os.path.join(daily_dir, f"TONG_HOP_{date_str}.txt")
  with open(sp, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
  print(f"[DAILY] {sp} ({len(imgs)} ảnh, {len(incidents)} sự kiện)")