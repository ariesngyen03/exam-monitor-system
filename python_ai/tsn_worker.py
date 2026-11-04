import collections
import queue
import threading
import time
import numpy as np

import ai_core
from ai_core import Config


class TSNWorker:
    """Quản lý mô hình TSN/3DCNN (MMAction2) và luồng nền trích xuất đặc trưng hành vi
    cho từng chỗ ngồi từ chuỗi buffer frame."""

    def __init__(self, cfg=Config, buffer_len=None):
        self.cfg = cfg
        self.buffer_len = buffer_len or getattr(cfg, "CNN3D_CLIP_LEN", 40)
        self.ready = False
        self._model_bundle = None

        self._queue = queue.Queue(maxsize=1)
        self._buffers = collections.defaultdict(lambda: collections.deque(maxlen=self.buffer_len))
        self._features = {}       # seat_idx -> np.ndarray (8,)
        self._last_submit = {}    # seat_idx -> timestamp của lần gửi gần nhất
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        """Khởi động luồng nền nạp model và chạy suy luận bất đồng bộ."""
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="TSNWorker")
        self._thread.start()
        return True

    def stop(self):
        """Dừng luồng nền."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def push_frame(self, seat_idx, person_crop):
        """Đưa crop ảnh người của chỗ ngồi tương ứng vào buffer và gửi task khi đủ frame."""
        if person_crop is None or person_crop.size == 0:
            return

        buf = self._buffers[seat_idx]
        buf.append(person_crop)

        # Gửi clip khi buffer đã đầy và đã qua ít nhất 1.0s kể từ lần phân tích trước của ghế này
        now = time.time()
        last_t = self._last_submit.get(seat_idx, 0.0)
        if len(buf) >= self.buffer_len and self.ready and (now - last_t >= 1.0):
            try:
                self._queue.put_nowait((seat_idx, list(buf)))
                self._last_submit[seat_idx] = now
            except queue.Full:
                pass

    def get_features(self, seat_idx):
        """Lấy vector đặc trưng 8 chiều mới nhất của chỗ ngồi."""
        with self._lock:
            return self._features.get(seat_idx, None)

    def clear(self, seat_idx):
        """Xóa buffer và dữ liệu khi chỗ ngồi trống."""
        with self._lock:
            self._buffers.pop(seat_idx, None)
            self._features.pop(seat_idx, None)
            self._last_submit.pop(seat_idx, None)

    def _run_loop(self):
        """Vòng lặp nạp model MMAction2 và xử lý clip bất đồng bộ."""
        try:
            model_bundle, _ = ai_core.init_3dcnn_model(self.cfg)
            if model_bundle is None:
                print("[TSNWorker][WARN] Không nạp được mô hình TSN/3DCNN.")
                self.ready = False
                return
            self._model_bundle = model_bundle
            self.ready = True
            print("[TSNWorker] Đã nạp xong TSN/3DCNN trong luồng nền — sẵn sàng phân tích hành vi.")
        except Exception as e:
            print(f"[TSNWorker][ERROR] Khởi tạo TSN thất bại: {e!r}")
            self.ready = False
            return

        while not self._stop_event.is_set():
            try:
                seat_idx, clip = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                feat = ai_core.extract_3dcnn_clip_features(self._model_bundle, None, clip, cfg=self.cfg)
                if feat is not None:
                    with self._lock:
                        self._features[seat_idx] = feat
            except Exception as e:
                print(f"[TSNWorker][WARN] Lỗi khi trích xuất đặc trưng ghế {seat_idx}: {e!r}")
