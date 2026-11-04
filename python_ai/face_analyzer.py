"""
face_analyzer.py — Tìm mặt TRONG vùng đầu của từng người đã dò được.

Vì sao không chạy Face Mesh trên cả khung hình: video cả lớp quay từ trên cao có mặt chỉ vài chục pixel,
Face Mesh cả khung hình dò được 0 mặt. Ở đây cắt vùng đầu từ ẢNH GỐC (độ phân giải đầy đủ), phóng to rồi mới
đưa vào Face Mesh (chế độ ảnh tĩnh), sau đó quy toạ độ về khung hình gốc.
Trả về None nếu không thấy mặt: nơi gọi khi đó KHÔNG chạy nhận diện cảm xúc.
"""
import cv2

import ai_core


def _g(cfg, k, d):
    return getattr(cfg, k, d) if cfg is not None else d


class HaarFaceFinder:
    """Dò mặt bằng Haar Cascade (bộ dò của ứng dụng nhận diện cảm xúc gốc) — PHƯƠNG ÁN DỰ PHÒNG chỉ cho nhánh
    CẢM XÚC, dùng khi Face Mesh không dò được mặt (mặt nhỏ, hơi nghiêng, ánh sáng xấu).
    Thông số giữ đúng bản gốc: scaleFactor=1.3, minNeighbors=5, margin 10%. Khác bản gốc: chỉ dò trong VÙNG ĐẦU
    của người đã dò được; nhiều mặt thì lấy mặt LỚN NHẤT; kích thước mặt tối thiểu HAAR_MIN_FACE mặc định 48 px
    (bản gốc 80; 48 là cỡ ảnh đầu vào mô hình cảm xúc DeepFace). Kết quả CHỈ dùng để lấy cảm xúc, không dùng cho
    yaw/pitch/ear/gaze nên vector đặc trưng và luật hành vi không đổi."""
    def __init__(self, cfg=None):
        self.min_face = _g(cfg, "HAAR_MIN_FACE", 48)
        self.margin = _g(cfg, "FACE_MARGIN_RATIO", 0.1)
        self.target_side = _g(cfg, "FACE_CROP_TARGET_SIDE", 320)
        self.max_scale = _g(cfg, "FACE_CROP_MAX_SCALE", 6.0)
        self.cascade = None
        try:
            c = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            if not c.empty():
                self.cascade = c
        except Exception:
            self.cascade = None
        self.ready = self.cascade is not None

    def find(self, frame, region):
        """region=(x0,y0,x1,y1) trong khung hình gốc. Trả về {"img": ảnh mặt đã cắt (có margin), "bbox": (x0,y0,x1,y1)
        trong khung hình gốc} hoặc None."""
        if not self.ready:
            return None
        hx0, hy0, hx1, hy1 = [int(v) for v in region]
        crop = frame[hy0:hy1, hx0:hx1]
        if crop.shape[0] < 12 or crop.shape[1] < 12:
            return None
        s = min(self.max_scale, max(1.0, self.target_side / float(max(crop.shape[:2]))))
        up = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC) if s > 1.05 else crop
        gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
        m = max(8, int(self.min_face * s))
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(m, m))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = int(x), int(y), int(w), int(h)
        mg = int(w * self.margin)
        y1, y2 = max(0, y - mg), min(up.shape[0], y + h + mg)
        x1, x2 = max(0, x - mg), min(up.shape[1], x + w + mg)
        img = up[y1:y2, x1:x2].copy()
        if img.size == 0:
            return None
        return {"img": img, "bbox": (hx0 + x / s, hy0 + y / s, hx0 + (x + w) / s, hy0 + (y + h) / s)}


class FaceAnalyzer:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.head_frac = _g(cfg, "FACE_HEAD_FRAC", 0.45)
        self.target_side = _g(cfg, "FACE_CROP_TARGET_SIDE", 320)
        self.max_scale = _g(cfg, "FACE_CROP_MAX_SCALE", 6.0)
        self.mesh = ai_core.mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=_g(cfg, "FACE_CROP_MIN_CONF", 0.3))
        # Dự phòng CHỈ cho cảm xúc (bộ dò Haar của ứng dụng cảm xúc gốc) — xem HaarFaceFinder ở trên
        self.haar = HaarFaceFinder(cfg) if _g(cfg, "EMOTION_HAAR_FALLBACK", True) else None

    def head_region(self, bbox, frame_shape):
        H, W = frame_shape[:2]
        x0, y0, x1, y1 = bbox
        w, h = x1 - x0, y1 - y0
        hx0, hx1 = max(0, int(x0 - 0.15 * w)), min(W, int(x1 + 0.15 * w))
        hy0, hy1 = max(0, int(y0 - 0.05 * h)), min(H, int(y0 + self.head_frac * h))
        return hx0, hy0, hx1, hy1

    def analyze(self, frame, bbox):
        hx0, hy0, hx1, hy1 = self.head_region(bbox, frame.shape)
        crop = frame[hy0:hy1, hx0:hx1]
        if crop.shape[0] < 12 or crop.shape[1] < 12:
            return None
        s = min(self.max_scale, max(1.0, self.target_side / float(max(crop.shape[:2]))))
        up = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC) if s > 1.05 else crop
        ch, cw = up.shape[:2]
        res = self.mesh.process(cv2.cvtColor(up, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None
        lms = res.multi_face_landmarks[0].landmark
        xs = [p.x * cw for p in lms]
        ys = [p.y * ch for p in lms]
        fx0, fx1, fy0, fy1 = min(xs), max(xs), min(ys), max(ys)
        if not ai_core._is_plausible_face(lms, fx0, fy0, fx1, fy1):
            return None
        yaw, pitch, roll = ai_core.compute_head_pose(lms, cw, ch)
        ear_l = ai_core.compute_ear(lms, ai_core.LEFT_EYE_IDX, cw, ch)
        ear_r = ai_core.compute_ear(lms, ai_core.RIGHT_EYE_IDX, cw, ch)
        gaze = (ai_core.compute_gaze_ratio_iris(lms, ai_core.RIGHT_EYE_IDX, ai_core.RIGHT_IRIS_IDX, cw, ch)
                if len(lms) >= 478 else 0.5)
        m = 0.1
        ix0, iy0 = max(0, int(fx0 - m * (fx1 - fx0))), max(0, int(fy0 - m * (fy1 - fy0)))
        ix1, iy1 = min(cw, int(fx1 + m * (fx1 - fx0))), min(ch, int(fy1 + m * (fy1 - fy0)))
        img = up[iy0:iy1, ix0:ix1]
        return {"yaw": float(yaw), "pitch": float(pitch), "roll": float(roll),
                "ear_l": float(ear_l), "ear_r": float(ear_r), "gaze": float(gaze),
                "bbox": (hx0 + fx0 / s, hy0 + fy0 / s, hx0 + fx1 / s, hy0 + fy1 / s),
                "img": img if img.size else None}

    def haar_face(self, frame, bbox):
        """Mặt dự phòng cho nhánh cảm xúc khi Face Mesh không dò được. Trả về {"img","bbox"} hoặc None."""
        if self.haar is None or not self.haar.ready:
            return None
        return self.haar.find(frame, self.head_region(bbox, frame.shape))