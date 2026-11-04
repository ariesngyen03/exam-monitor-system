"""
emotion_worker.py — Chạy suy luận NHẬN DIỆN CẢM XÚC (DeepFace) trong LUỒNG
NỀN riêng biệt, kiến trúc giống hệt tsn_worker.py.

VÌ SAO CẦN FILE NÀY:
DeepFace (chạy trên TensorFlow) khá nặng — mỗi lần phân tích 1 khuôn mặt có
thể mất hàng trăm mili-giây. Nếu chạy trực tiếp trong vòng lặp camera
chính, hình ảnh sẽ bị khựng/lag — đúng lý do TSN (3DCNN) phải chạy nền.

KHÁC BIỆT SO VỚI BỘ CODE GỐC (AI_test — detector.py dùng Haar Cascade):
Hệ thống chính ĐÃ CÓ SẴN khuôn mặt của từng người từ MediaPipe Face Mesh
(process_faces_frame() trong ai_core.py) — vì vậy KHÔNG dò mặt lại lần thứ
2 bằng Haar Cascade nữa (vừa tốn tài nguyên, vừa có thể ra 2 khung bao lệch
nhau giữa 2 bộ dò khác nhau). Thay vào đó, chỉ cần CẮT đúng vùng khuôn mặt
đã có sẵn (ai_core.crop_face_region()) rồi gửi thẳng cho DeepFace phân
tích với detector_backend="skip" (bỏ qua bước dò mặt nội bộ của DeepFace).

ĐA NGƯỜI: mỗi người có 1 cửa sổ trượt (sliding window) lịch sử cảm xúc
riêng, làm mượt bằng phép lấy giá trị xuất hiện nhiều nhất (mode) — giữ
đúng logic làm mượt của bản gốc (thư viện `statistics.mode`). Việc gửi ảnh
vào luồng nền được xoay vòng (round-robin) — mỗi khung hình chỉ gửi ĐÚNG 1
người — để không quá tải khi giám sát nhiều người cùng lúc.
"""
import collections
import logging
import queue
import statistics
import threading
import time

logger = logging.getLogger(__name__)

DEFAULT_EMOTION_LABEL = "neutral"
DARK_IMAGE_LABEL = "khong ro (anh toi)"


class EmotionWorker:
    """Quản lý DeepFace + luồng nền phân tích cảm xúc cho từng người."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.ready = False
        self._deepface = None

        self._queue = queue.Queue(maxsize=1)   # chỉ giữ ĐÚNG 1 yêu cầu mới nhất — không dồn việc
        self._history = {}   # person_id -> deque lịch sử cảm xúc (cửa sổ trượt)
        self._emotions = {}  # person_id -> nhãn cảm xúc ĐANG HIỂN THỊ (đã giữ ổn định, xem _hold_update)
        self._pending = {}   # person_id -> (nhãn ứng viên khác nhãn đang hiển thị, lúc bắt đầu thấy nó)
        self._smoothed = {}  # person_id -> nhãn đã làm mượt bằng mode, CHƯA qua bước giữ ổn định (đúng như bản gốc)
        self._scores = {}    # person_id -> dict điểm số chi tiết từng cảm xúc (tuỳ chọn, cho UI/log)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        """Khởi động luồng nền NGAY LẬP TỨC — KHÔNG đợi import deepface
        xong. Việc `from deepface import DeepFace` (kéo theo TensorFlow,
        khá nặng, có thể mất vài giây) diễn ra BÊN TRONG luồng nền
        (_run_loop), nên hàm này trả về gần như tức thời, không chặn luồng
        camera chính. Luôn trả về True (đã khởi động luồng) — muốn biết
        nhánh cảm xúc đã THỰC SỰ sẵn sàng hay chưa, kiểm tra self.ready
        (cập nhật bất đồng bộ)."""
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="EmotionWorker")
        self._thread.start()
        return True

    def stop(self):
        """Dừng luồng nền êm — gọi khi kết thúc phiên giám sát."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def submit_face(self, person_id, face_crop):
        """Gửi 1 ảnh khuôn mặt ĐÃ CẮT SẴN vào hàng đợi xử lý.

        KHÔNG BAO GIỜ chặn luồng gọi (luồng camera chính) — nếu hàng đợi
        đang bận (ảnh trước chưa xử lý xong), lượt này bị BỎ QUA, đợi lượt
        xoay vòng kế tiếp thay vì xếp hàng chờ."""
        if not self.ready or face_crop is None or face_crop.size == 0:
            return
        try:
            self._queue.put_nowait((person_id, face_crop))
        except queue.Full:
            pass
    def get_emotion(self, person_id, default=DEFAULT_EMOTION_LABEL):
        """Nhãn cảm xúc MỚI NHẤT đã làm mượt; trả về `default` nếu chưa phân tích được
        (truyền default=None để phân biệt 'chưa có kết quả' với 'neutral')."""
        with self._lock:
            return self._emotions.get(person_id, default)

    def get_smoothed(self, person_id, default=None):
        """Nhãn cảm xúc đã làm mượt bằng mode (đúng như bản gốc), KHÔNG qua bước giữ ổn định 5 giây.
        Dùng để tính cờ 'lo lắng/căng thẳng kéo dài' đúng theo thuật toán gốc."""
        with self._lock:
            return self._smoothed.get(person_id, default)

    def clear(self, person_id):
        """Xoá cảm xúc + lịch sử của 1 người (gọi khi mất mặt để không hiển thị kết quả cũ)."""
        with self._lock:
            self._emotions.pop(person_id, None)
            self._scores.pop(person_id, None)
            self._history.pop(person_id, None)
            self._pending.pop(person_id, None)
            self._smoothed.pop(person_id, None)

    def _hold_update(self, person_id, candidate):
        """Chỉ đổi cảm xúc ĐANG HIỂN THỊ khi ứng viên mới xuất hiện ỔN ĐỊNH đủ EMOTION_HOLD_SECONDS liên
        tục — tránh nhảy qua lại giữa các cảm xúc gần ngang nhau mỗi lần phân tích (DeepFace vốn khá nhạy
        cảm với biểu cảm thoáng qua). PHẢI gọi trong lúc đã giữ self._lock."""
        hold = getattr(self.cfg, "EMOTION_HOLD_SECONDS", 5.0)
        if hold <= 0:                                      # EMOTION_HOLD_SECONDS = 0 -> đúng hành vi bản gốc (đổi ngay)
            self._emotions[person_id] = candidate
            self._pending.pop(person_id, None)
            return
        now = time.time()
        current = self._emotions.get(person_id)
        if current is None or candidate == current:
            self._emotions[person_id] = candidate
            self._pending.pop(person_id, None)
            return
        cand, since = self._pending.get(person_id, (None, None))
        if cand != candidate:
            self._pending[person_id] = (candidate, now)   # ứng viên MỚI (khác lần trước) -> đếm lại từ đầu
        elif now - since >= hold:
            self._emotions[person_id] = candidate          # đã thấy ỔN ĐỊNH đủ lâu -> mới thật sự đổi hiển thị
            self._pending.pop(person_id, None)

    def get_emotion_scores(self, person_id):
        """Đọc điểm số chi tiết từng loại cảm xúc của lần phân tích gần nhất
        (VD dùng cho biểu đồ/log chi tiết) — trả về None nếu chưa có."""
        with self._lock:
            return self._scores.get(person_id)

    def _run_loop(self):
        """Vòng lặp của luồng nền — chạy độc lập hoàn toàn với luồng camera.
        BƯỚC ĐẦU TIÊN của luồng này mới là lúc thực sự import deepface
        (trước đây nằm trong start(), nay chuyển hẳn vào đây)."""
        try:
            from deepface import DeepFace
            self._deepface = DeepFace
        except ImportError as e:
            print(f"[EmotionWorker][WARN] Chưa cài deepface ({e!r}) — nhánh cảm xúc sẽ KHÔNG chạy.")
            self.ready = False
            return   # import thất bại -> thoát hẳn luồng, không có gì để xử lý tiếp

        self.ready = True
        print("[EmotionWorker] Đã nạp xong deepface trong luồng nền — sẵn sàng phân tích cảm xúc.")

        window_size = getattr(self.cfg, "EMOTION_SLIDING_WINDOW_SIZE", 7)
        while not self._stop_event.is_set():
            try:
                person_id, face_crop = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                # SỬA (khắc phục cảm xúc "dính" 1 nhãn sai khi phòng tối —
                # thấy rõ qua ảnh thực tế, nhãn "sad" không đổi trong ảnh
                # gần như tối đen): DeepFace vẫn cố "đoán" 1 cảm xúc dù ảnh
                # thiếu sáng nghiêm trọng, kết quả không đáng tin (thường
                # rơi vào 1 nhãn cố định do đặc trưng ảnh tối làm model
                # nhầm lẫn). Kiểm tra ĐỘ SÁNG TRUNG BÌNH của ảnh khuôn mặt
                # TRƯỚC khi đưa vào DeepFace — nếu quá tối, KHÔNG đoán bừa,
                # trả về nhãn RÕ RÀNG LÀ "không xác định" thay vì 1 cảm xúc
                # cụ thể trông có vẻ đáng tin nhưng thực ra vô nghĩa.
                min_brightness = getattr(self.cfg, "EMOTION_MIN_BRIGHTNESS", 40)
                if face_crop.mean() < min_brightness:
                    with self._lock:
                        self._emotions[person_id] = DARK_IMAGE_LABEL
                        self._smoothed[person_id] = DARK_IMAGE_LABEL
                        self._scores[person_id] = None
                    continue

                result = self._deepface.analyze(
                    face_crop,
                    actions=["emotion"],
                    enforce_detection=False,   # ảnh đã chắc chắn có mặt (MediaPipe đã dò trước) -> bỏ qua kiểm tra lại
                    detector_backend="skip",   # không dò mặt lại lần 2 bên trong DeepFace
                )
                emotion = result[0]["dominant_emotion"]
                scores = result[0].get("emotion")

                hist = self._history.setdefault(person_id, collections.deque(maxlen=window_size))
                hist.append(emotion)
                smoothed = statistics.mode(hist)

                with self._lock:
                    self._smoothed[person_id] = smoothed
                    self._hold_update(person_id, smoothed)
                    self._scores[person_id] = scores
            except Exception:
                logger.exception("[EmotionWorker] Lỗi phân tích cảm xúc người #%s", person_id)