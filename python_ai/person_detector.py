"""
person_detector.py — Dò người (toàn thân) cho camera trên cao. Chạy ĐỒNG BỘ (gọi từ luồng phân tích).

Backend theo thứ tự ưu tiên (PERSON_BACKEND="auto"):
  1. OpenVINO: cùng trọng số yolov8s đã xuất sang OpenVINO IR (tools/export_openvino.py). Đo trên i5-12500H,
     khung 1080p: ~41 ms/lần so với ~200 ms của PyTorch, số người dò được như nhau. Cài: pip install openvino
  2. ultralytics YOLO (PyTorch). Cài: pip install ultralytics
  3. torchvision Faster R-CNN (dự phòng, chậm trên CPU).

Kết quả luôn được lọc: bỏ box quá nhỏ/quá lớn, bỏ trùng lặp (NMS), bỏ box nằm gọn trong box khác.
"""
import os

import cv2
import numpy as np


def _g(cfg, k, d):
    return getattr(cfg, k, d) if cfg is not None else d


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def contained(inner, outer, frac=0.85):
    ix0, iy0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix1, iy1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return area > 0 and inter / area >= frac


def filter_dets(dets, frame_w, frame_h, cfg=None):
    """dets: list {"bbox":(x0,y0,x1,y1), "conf":float}. Trả về list đã lọc, sắp theo conf giảm dần."""
    min_area = _g(cfg, "PERSON_MIN_AREA_RATIO", 0.002) * frame_w * frame_h
    max_area = _g(cfg, "PERSON_MAX_AREA_RATIO", 0.25) * frame_w * frame_h
    nms_iou = _g(cfg, "PERSON_NMS_IOU", 0.5)
    keep = []
    for d in sorted(dets, key=lambda d: -d["conf"]):
        x0, y0, x1, y1 = d["bbox"]
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        if area < min_area or area > max_area:
            continue
        if any(iou(d["bbox"], k["bbox"]) > nms_iou or contained(d["bbox"], k["bbox"]) for k in keep):
            continue
        keep.append(d)
    return keep


class OpenVinoYolo:
    """YOLOv8 dạng OpenVINO IR (kích thước vào cố định, VD 544x960 cho khung 16:9), chạy thẳng bằng openvino thay
    vì qua ultralytics để tự chọn thiết bị và giới hạn số luồng CPU.
    Thiết bị "auto": GPU tích hợp Intel nếu có (Iris Xe ~30 ms/lần, KHÔNG tốn CPU -> camera + giao diện C++ chạy cùng
    máy không bị giật), ngược lại CPU. Không dùng thiết bị "AUTO" của OpenVINO: đo được chậm hơn GPU 4 lần.
    Tiền/hậu xử lý giống ultralytics: letterbox nền 114, RGB/255; giữ box có lớp mạnh nhất là người; NMS IoU 0.7."""

    def __init__(self, xml_path, threads=0, device="auto"):
        import openvino as ov
        core = ov.Core()
        # Biên dịch cho GPU mất vài giây — lưu bản đã biên dịch để các lần chạy sau mở ngay
        core.set_property({"CACHE_DIR": os.path.join(os.path.dirname(xml_path), "ov_cache")})
        if threads > 0:
            core.set_property("CPU", {"INFERENCE_NUM_THREADS": int(threads)})
        model = core.read_model(xml_path)
        _, _, self.in_h, self.in_w = (int(d) for d in model.input(0).shape)
        if device == "auto":
            candidates = (["GPU"] if any(d.startswith("GPU") for d in core.available_devices) else []) + ["CPU"]
        else:
            candidates = [device.upper()]
        for i, dev in enumerate(candidates):
            try:
                self.compiled = core.compile_model(model, dev, {"PERFORMANCE_HINT": "LATENCY"})
                self.device = f"{dev} ({core.get_property(dev, 'FULL_DEVICE_NAME')})"
                break
            except Exception as e:                       # VD driver GPU cũ -> thử thiết bị kế tiếp
                if i == len(candidates) - 1:
                    raise
                print(f"[PersonDetector][WARN] OpenVINO không dùng được {dev}: {e!r}")

    def predict(self, frame_bgr, conf):
        h, w = frame_bgr.shape[:2]
        r = min(self.in_h / h, self.in_w / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        left, top = (self.in_w - nw) // 2, (self.in_h - nh) // 2
        canvas = np.full((self.in_h, self.in_w, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = (
            cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else frame_bgr)
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255.0, swapRB=True)
        pred = self.compiled([blob])[0][0]                 # (4 + số lớp, số anchor): cx, cy, w, h, điểm từng lớp
        score = pred[4]
        keep = (score > conf) & (pred[4:].argmax(0) == 0)
        if not keep.any():
            return []
        cx, cy, bw, bh = pred[0, keep], pred[1, keep], pred[2, keep], pred[3, keep]
        score = score[keep]
        x0 = np.clip((cx - bw / 2 - left) / r, 0, w)
        y0 = np.clip((cy - bh / 2 - top) / r, 0, h)
        x1 = np.clip((cx + bw / 2 - left) / r, 0, w)
        y1 = np.clip((cy + bh / 2 - top) / r, 0, h)
        rects = np.stack([x0, y0, x1 - x0, y1 - y0], axis=1).tolist()
        idx = cv2.dnn.NMSBoxes(rects, score.tolist(), conf, 0.7)
        return [{"bbox": (float(x0[i]), float(y0[i]), float(x1[i]), float(y1[i])), "conf": float(score[i])}
                for i in np.asarray(idx).reshape(-1)]


class PersonDetector:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.ready = False
        self.backend = None
        self._model = None
        self._load()

    def _load(self):
        want = _g(self.cfg, "PERSON_BACKEND", "auto")
        errors = []
        ov_xml = _g(self.cfg, "PERSON_OPENVINO_MODEL", "")
        if want in ("auto", "openvino") and ov_xml:
            try:
                if not os.path.isfile(ov_xml):
                    raise FileNotFoundError(f"{ov_xml} (tạo bằng python_ai/tools/export_openvino.py)")
                self._model = OpenVinoYolo(ov_xml, _g(self.cfg, "PERSON_THREADS", 0), _g(self.cfg, "PERSON_DEVICE", "auto"))
                self.backend = "openvino"
            except Exception as e:                       # chưa cài openvino hoặc chưa xuất mô hình
                errors.append(f"openvino: {e!r}")
        if self._model is None and want in ("auto", "yolo"):
            try:
                from ultralytics import YOLO
                self._model = YOLO(_g(self.cfg, "PERSON_YOLO_MODEL", "yolov8n.pt"))
                self.backend = "yolo"
            except Exception as e:                       # chưa cài ultralytics hoặc không tải được trọng số
                errors.append(f"ultralytics: {e!r}")
        if self._model is None and want in ("auto", "torchvision"):
            try:
                import torch
                from torchvision.models.detection import (
                    fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights)
                self._torch = torch
                self._model = fasterrcnn_mobilenet_v3_large_fpn(
                    weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT).eval()
                self.backend = "torchvision"
            except Exception as e:
                errors.append(f"torchvision: {e!r}")
        self.ready = self._model is not None
        if self.ready:
            print(f"[PersonDetector] Backend: {self.backend}"
                  + (f" trên {self._model.device}" if self.backend == "openvino" else ""))
            if self.backend != "openvino" and errors:
                print("[PersonDetector][WARN] Không dùng được OpenVINO (chậm hơn ~5 lần): " + " | ".join(errors))
            if self.backend == "torchvision":
                print("[PersonDetector][WARN] Backend torchvision chậm trên CPU; nên cài ultralytics: pip install ultralytics")
        else:
            print("[PersonDetector][LỖI] Không nạp được detector người: " + " | ".join(errors))

    def detect(self, frame_bgr):
        if not self.ready:
            return []
        h, w = frame_bgr.shape[:2]
        conf = _g(self.cfg, "PERSON_CONF", 0.35)
        raw = []
        if self.backend == "openvino":
            raw = self._model.predict(frame_bgr, conf)
        elif self.backend == "yolo":
            r = self._model.predict(frame_bgr, imgsz=_g(self.cfg, "PERSON_IMGSZ", 960),
                                    conf=conf, classes=[0], verbose=False)[0]
            xyxy = r.boxes.xyxy.cpu().numpy()
            cs = r.boxes.conf.cpu().numpy()
            raw = [{"bbox": tuple(float(v) for v in b), "conf": float(c)} for b, c in zip(xyxy, cs)]
        else:
            torch = self._torch
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
            with torch.no_grad():
                out = self._model([t])[0]
            for b, l, s in zip(out["boxes"].numpy(), out["labels"].numpy(), out["scores"].numpy()):
                if int(l) == 1 and float(s) >= conf:
                    raw.append({"bbox": tuple(float(v) for v in b), "conf": float(s)})
        return filter_dets(raw, w, h, self.cfg)
