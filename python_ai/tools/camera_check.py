"""
camera_check.py — Kiểm tra camera trên máy chạy thử: định dạng (MJPG/YUY2) và số frame/giây THỰC TẾ.

Giao diện C++ đọc camera qua DirectShow (bản OpenCV tự build không có MSMF). Thứ tự đặt thuộc tính quyết định định dạng:
  cũ:  FOURCC -> rộng -> cao -> FPS   => OpenCV quên MJPG, chọn YUY2 1080p (USB 2.0 chỉ ~5-10 fps)
  mới: FPS -> rộng -> cao -> FOURCC   => MJPG 1080p @ 30 fps
Công cụ chạy cả hai cách trên cùng camera để so sánh.

    python python_ai/tools/camera_check.py --index 0
"""
import argparse
import sys
import time

import cv2

MJPG = cv2.VideoWriter_fourcc(*"MJPG")


def fourcc_str(v):
    v = int(v)
    return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4)) if v > 0 else "?"


def measure(index, width, height, order, seconds):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return f"không mở được camera {index}"
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    steps = {"FOURCC": (cv2.CAP_PROP_FOURCC, MJPG), "W": (cv2.CAP_PROP_FRAME_WIDTH, width),
             "H": (cv2.CAP_PROP_FRAME_HEIGHT, height), "FPS": (cv2.CAP_PROP_FPS, 30)}
    for name in order:
        cap.set(*steps[name])
    for _ in range(5):                       # bỏ vài frame đầu (camera đang tự chỉnh sáng)
        cap.read()
    n, t0, shape = 0, time.perf_counter(), None
    while time.perf_counter() - t0 < seconds:
        ok, frame = cap.read()
        if ok:
            n += 1
            shape = frame.shape
    fps = n / (time.perf_counter() - t0)
    fmt = fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()
    size = f"{shape[1]}x{shape[0]}" if shape else "không có frame"
    return f"{size} {fmt} -> {fps:.1f} fps thực tế"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--seconds", type=float, default=5.0)
    a = ap.parse_args()
    print(f"OpenCV {cv2.__version__}, camera {a.index}, yêu cầu {a.width}x{a.height} MJPG @ 30 fps (DirectShow)")
    for label, order in (("Thứ tự CŨ  (FOURCC, W, H, FPS)", ["FOURCC", "W", "H", "FPS"]),
                         ("Thứ tự MỚI (FPS, W, H, FOURCC)", ["FPS", "W", "H", "FOURCC"])):
        print(f"  {label}: {measure(a.index, a.width, a.height, order, a.seconds)}", flush=True)
    print("MJPG ~25-30 fps là đúng. Nếu MJPG mà vẫn thấp: phòng tối (camera tự giảm fps để phơi sáng lâu hơn).")


if __name__ == "__main__":
    main()
