"""
mock_cpp_client.py — Giả lập C++ Client để kiểm thử AI Server mà không cần build Qt.

Đọc video/camera, gửi FramePacket (JPEG) qua PUB tcp://*:5555 như ZmqPublisher, nhận InferenceResult
từ tcp://127.0.0.1:5556 như ZmqSubscriber và in tóm tắt kết quả.

    python python_ai/main.py --no-emotion            # cửa sổ 1
    python python_ai/tools/mock_cpp_client.py --source video.mp4 --seconds 40   # cửa sổ 2
"""
import argparse
import os
import sys
import time

import cv2
import zmq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import pb  # noqa: E402  (dùng chung cách nạp proto của server)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="Đường dẫn video/ảnh hoặc chỉ số camera")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--fps", type=float, default=15.0)
    a = ap.parse_args()

    still = None
    if a.source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        still = cv2.imread(a.source)   # ảnh tĩnh: gửi lặp lại cùng 1 khung hình
        if still is None:
            sys.exit(f"Không đọc được ảnh: {a.source}")
    else:
        src = int(a.source) if a.source.isdigit() else a.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            sys.exit(f"Không mở được nguồn: {a.source}")

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 2)
    pub.bind("tcp://*:5555")
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.connect("tcp://127.0.0.1:5556")
    time.sleep(0.5)

    t_end, frame_id, n_res, n_hb, n_alert = time.time() + a.seconds, 0, 0, 0, 0
    last_print = 0.0
    while time.time() < t_end:
        if still is not None:
            frame = still.copy()
        else:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
        frame = cv2.resize(frame, (1280, 720))
        frame_id += 1
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        pkt = pb.FramePacket(timestamp=int(time.time() * 1000), frame_id=frame_id,
                             image_data=buf.tobytes(), width=1280, height=720)
        pub.send(pkt.SerializeToString(), zmq.DONTWAIT)

        while True:
            try:
                msg = sub.recv(zmq.NOBLOCK)
            except zmq.Again:
                break
            r = pb.InferenceResult()
            r.ParseFromString(msg)
            if r.is_heartbeat:
                n_hb += 1
                continue
            n_res += 1
            for d in r.detections:
                if d.new_alert:
                    n_alert += 1
                    print(f"  [ALERT] {d.seat_label} {d.behavior_type} {d.confidence:.0%} {list(d.violations)} "
                          f"-> {d.evidence_path}")
            if time.time() - last_print > 2.0:
                last_print = time.time()
                state = f"HỌC CHỖ {r.calib_progress:.0%}" if r.calibrating else "GIÁM SÁT"
                kinds = ", ".join(f"{d.seat_label or d.track_id}:{d.behavior_type}" for d in r.detections)
                print(f"[frame {r.frame_id}] {state} | người={r.n_people} mặt={r.n_faces} | {kinds}")
        time.sleep(1.0 / a.fps)

    print(f"\nTổng: {frame_id} frame gửi | {n_res} kết quả | {n_hb} heartbeat | {n_alert} cảnh báo")
    pub.close(0)
    sub.close(0)
    ctx.term()


if __name__ == "__main__":
    main()
