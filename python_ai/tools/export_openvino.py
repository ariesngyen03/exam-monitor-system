"""
export_openvino.py — Xuất yolov8s.pt sang OpenVINO IR cho PersonDetector (backend "openvino").

    pip install openvino
    python python_ai/tools/export_openvino.py            :: -> python_ai/models/yolov8s_openvino_544x960/

Kích thước vào cố định 544x960 (cao x rộng): khung 16:9 (1080p/720p) thu nhỏ cạnh dài còn 960 như PERSON_IMGSZ,
không tốn phép tính cho phần đệm vuông. Khung tỉ lệ khác vẫn chạy được (letterbox) nhưng ảnh bị thu nhỏ hơn.
"""
import argparse
import os
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(BASE_DIR, "models", "yolov8s.pt"))
    ap.add_argument("--height", type=int, default=544)
    ap.add_argument("--width", type=int, default=960)
    a = ap.parse_args()

    from ultralytics import YOLO
    stem = os.path.splitext(os.path.basename(a.weights))[0]
    out = YOLO(a.weights).export(format="openvino", imgsz=[a.height, a.width], half=False, dynamic=False)
    dst = os.path.join(BASE_DIR, "models", f"{stem}_openvino_{a.height}x{a.width}")
    if os.path.abspath(out) != os.path.abspath(dst):
        shutil.rmtree(dst, ignore_errors=True)
        shutil.move(out, dst)
    print(f"Đã xuất: {dst}")


if __name__ == "__main__":
    main()
