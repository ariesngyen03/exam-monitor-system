r"""
package.py — Đóng gói ExamMonitorSystem thành 1 thư mục chạy được trên máy Windows khác (không cần cài gì).

    scripts\package.bat                 :: = python scripts\package.py (đọc đường dẫn từ scripts\local_env.bat)
    python scripts\package.py --zip     :: thêm file .zip để gửi đi

Kết quả: dist\ExamMonitorSystem\
    Start.bat            khởi động AI (có tự khởi động lại) + giao diện; đóng giao diện là tắt AI
    app\                 ExamMonitorSystem.exe + Qt/OpenCV/ZeroMQ/protobuf DLL
    python\              Python 3.11 portable + thư viện AI (tách khỏi Python/user-site của máy đích)
    python_ai\  proto\python_out\   mã AI (giữ đúng cấu trúc thư mục như repo)
    data\                ảnh bằng chứng, báo cáo, log (gửi lại thư mục này để phân tích)

Yêu cầu: chạy SAU khi build C++ (build\ExamMonitorSystem.exe) và bằng Python đã cài đủ requirements (3.11).
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Thư viện Python có trong máy dev nhưng AI không dùng (bỏ để gói nhẹ hơn ~450 MB)
SKIP_PACKAGES = {"jax", "jaxlib", "polars", "_polars_runtime_32", "pip", "setuptools", "virtualenv",
                 "IPython", "ipykernel", "jupyter_client", "jupyter_core", "debugpy", "notebook", "pytest"}
# Thư viện nằm ở user-site của máy dev (%APPDATA%\Python\Python311\site-packages) mà AI cần
USER_SITE_PACKAGES = ["zmq", "pyzmq-*.dist-info", "pyzmq.libs", "psutil", "psutil-*.dist-info",
                      "packaging", "packaging-*.dist-info", "dateutil", "python_dateutil-*.dist-info",
                      "six.py", "six-*.dist-info", "colorama", "colorama-*.dist-info"]
# Runtime Visual C++ mà torch cần (máy mới cài Windows có thể chưa có)
VC_RUNTIME = ["msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "vcruntime140.dll", "vcruntime140_1.dll",
              "concrt140.dll", "vcomp140.dll"]


def log(msg):
    print(f"[package] {msg}", flush=True)


def env(name, default=""):
    return os.environ.get(name, default).strip().strip('"')


def dll_imports(path, objdump):
    out = subprocess.run([objdump, "-p", path], capture_output=True, text=True, errors="replace").stdout
    return re.findall(r"DLL Name: (\S+)", out)


def copy_dll_closure(app_dir, search_dirs, objdump):
    """Chép đệ quy mọi DLL mà exe/DLL trong app_dir cần, tìm theo đúng thứ tự PATH của scripts\\run.bat.
    DLL hệ thống (KERNEL32, msvcrt...) không có trong search_dirs nên tự bỏ qua."""
    lookup = {}
    for d in search_dirs:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith(".dll"):
                    lookup.setdefault(f.lower(), os.path.join(d, f))
    todo = [os.path.join(app_dir, f) for f in os.listdir(app_dir) if f.lower().endswith((".exe", ".dll"))]
    for sub in ("platforms", "styles", "imageformats"):
        p = os.path.join(app_dir, sub)
        if os.path.isdir(p):
            todo += [os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(".dll")]
    seen = {os.path.basename(p).lower() for p in todo}
    copied = 0
    while todo:
        for name in dll_imports(todo.pop(), objdump):
            key = name.lower()
            if key in seen or key not in lookup:
                continue
            seen.add(key)
            dst = os.path.join(app_dir, os.path.basename(lookup[key]))
            shutil.copy2(lookup[key], dst)
            todo.append(dst)
            copied += 1
    return copied


def build_app(out, a):
    exe = os.path.join(ROOT, "build", "ExamMonitorSystem.exe")
    if not os.path.isfile(exe):
        sys.exit("Chưa có build\\ExamMonitorSystem.exe — chạy scripts\\build.bat trước.")
    app = os.path.join(out, "app")
    os.makedirs(app)
    shutil.copy2(exe, app)

    qt_bin = os.path.join(a.qt_dir, "bin")
    msys_bin = os.path.join(a.msys_dir, "bin")
    cv_bin = os.path.join(a.opencv_dir, "bin")
    zmq_bin = os.path.join(a.zmq_dir, "bin")
    run_env = dict(os.environ, PATH=os.pathsep.join([qt_bin, msys_bin, cv_bin, zmq_bin, os.environ["PATH"]]))
    log("windeployqt ...")
    subprocess.run([os.path.join(qt_bin, "windeployqt.exe"), "--release", "--no-translations",
                    "--no-system-d3d-compiler", "--no-opengl-sw", "--dir", app, os.path.join(app, "ExamMonitorSystem.exe")],
                   check=True, env=run_env, stdout=subprocess.DEVNULL)
    # OpenCV nạp plugin (VD videoio_ffmpeg để mở file video) lúc chạy -> objdump không thấy, chép cả thư mục bin
    for f in os.listdir(cv_bin):
        if f.lower().endswith(".dll"):
            shutil.copy2(os.path.join(cv_bin, f), app)
    n = copy_dll_closure(app, [qt_bin, msys_bin, cv_bin, zmq_bin], os.path.join(msys_bin, "objdump.exe"))
    log(f"app: thêm {n} DLL phụ thuộc")


def build_python(out, a):
    src = os.path.dirname(a.python)
    dst = os.path.join(out, "python")
    site = os.path.join("Lib", "site-packages")

    def ignore(d, names):
        rel = os.path.relpath(d, src)
        skip = {"__pycache__"}
        if rel == ".":
            skip |= {"Doc", "Scripts", "Tools", "include", "libs", "NEWS.txt"}
        if rel == "Lib":
            skip |= {"test", "idlelib", "ensurepip"}
        if rel == site:
            skip |= {n for n in names if n.split("-")[0] in SKIP_PACKAGES or n in SKIP_PACKAGES}
        return [n for n in names if n in skip]

    log(f"python: chép {src} (bỏ thư viện không dùng) ...")
    shutil.copytree(src, dst, ignore=ignore)

    import glob
    user_site = os.path.join(os.environ.get("APPDATA", ""), "Python", f"Python{sys.version_info[0]}{sys.version_info[1]}",
                             "site-packages")
    for pat in USER_SITE_PACKAGES:
        for p in glob.glob(os.path.join(user_site, pat)):
            target = os.path.join(dst, site, os.path.basename(p))
            if os.path.exists(target):
                continue
            (shutil.copytree if os.path.isdir(p) else shutil.copy2)(p, target)
            log(f"python: + {os.path.basename(p)} (từ user-site)")
    sysdir = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    for f in VC_RUNTIME:
        if os.path.isfile(os.path.join(sysdir, f)) and not os.path.exists(os.path.join(dst, f)):
            shutil.copy2(os.path.join(sysdir, f), dst)


def build_ai(out):
    ign = shutil.ignore_patterns("__pycache__", "*.pyc", "ov_cache")   # ov_cache: bản biên dịch riêng của máy dev
    shutil.copytree(os.path.join(ROOT, "python_ai"), os.path.join(out, "python_ai"), ignore=ign)
    shutil.copytree(os.path.join(ROOT, "proto", "python_out"), os.path.join(out, "proto", "python_out"), ignore=ign)
    for sub in ("evidence", "reports", "logs"):
        os.makedirs(os.path.join(out, "data", sub), exist_ok=True)


def write_launchers(out):
    pkg = os.path.join(ROOT, "scripts", "package_files")
    for f in os.listdir(pkg):
        shutil.copy2(os.path.join(pkg, f), out)


def smoke_test(out):
    """Nạp toàn bộ thư viện AI bằng Python trong gói, tách hẳn user-site / PYTHONPATH của máy này."""
    py = os.path.join(out, "python", "python.exe")
    # Chạy thật YOLO + Face Mesh để lộ cả các import lười (chỉ nạp lúc suy luận)
    code = ("import sys, numpy as np; sys.argv = ['main.py']; sys.path.insert(0, r'%s');"
            "import main; main.load_proto();"
            "import zmq, torch, mediapipe as mp, exam_ai_service, session;"
            "from ultralytics import YOLO; img = np.zeros((480, 640, 3), np.uint8);"
            "YOLO(r'%s').predict(img, verbose=False);"
            "mp.solutions.face_mesh.FaceMesh(refine_landmarks=True).process(img);"
            "from ai_core import Config; from person_detector import PersonDetector; d = PersonDetector(Config);"
            "d.detect(img); assert d.backend == 'openvino', 'thieu openvino: pip install openvino';"
            "print('OK', sys.version.split()[0], 'torch', torch.__version__, 'zmq', zmq.__version__, 'yolo', d._model.device)"
            ) % (os.path.join(out, "python_ai"), os.path.join(out, "python_ai", "models", "yolov8s.pt"))
    clean = {k: v for k, v in os.environ.items() if not k.upper().startswith("PYTHON")}
    clean["PATH"] = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    r = subprocess.run([py, "-s", "-c", code], capture_output=True, text=True, env=clean, errors="replace",
                       cwd=os.path.join(out, "python_ai"))
    if r.returncode != 0 or "OK" not in r.stdout:
        print(r.stdout, r.stderr[-3000:])
        sys.exit("Kiểm tra Python trong gói THẤT BẠI")
    log("kiểm tra Python trong gói: " + r.stdout.strip().splitlines()[-1])
    # Bản biên dịch OpenVINO vừa sinh là của máy này — máy chạy thử tự biên dịch lại lần đầu
    for d, subdirs, _ in os.walk(os.path.join(out, "python_ai", "models")):
        if "ov_cache" in subdirs:
            shutil.rmtree(os.path.join(d, "ov_cache"), ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "dist", "ExamMonitorSystem"))
    ap.add_argument("--zip", action="store_true", help="Nén thêm thành dist\\ExamMonitorSystem_<ngày>.zip")
    ap.add_argument("--qt-dir", default=env("LOCAL_QT_DIR", r"C:\Qt\6.11.1\mingw_64"))
    ap.add_argument("--msys-dir", default=env("LOCAL_MSYS2_DIR", r"C:\msys64\mingw64"))
    ap.add_argument("--opencv-dir", default=env("LOCAL_OPENCV_DIR"))
    ap.add_argument("--zmq-dir", default=env("LOCAL_ZMQ_DIR"))
    ap.add_argument("--python", default=env("LOCAL_AI_PYTHON") or sys.executable,
                    help="python.exe 3.10/3.11 đã cài đủ python_ai/requirements.txt")
    a = ap.parse_args()

    t0 = time.time()
    if os.path.exists(a.out):
        log(f"xoá bản cũ {a.out}")
        shutil.rmtree(a.out)
    os.makedirs(a.out)
    build_app(a.out, a)
    build_python(a.out, a)
    build_ai(a.out)
    write_launchers(a.out)
    smoke_test(a.out)

    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(a.out) for f in fs)
    log(f"xong: {a.out} ({size / 2**30:.2f} GB, {time.time() - t0:.0f} s)")
    if a.zip:
        base = os.path.join(os.path.dirname(a.out), f"ExamMonitorSystem_{time.strftime('%Y%m%d')}")
        log("nén zip ...")
        shutil.make_archive(base, "zip", os.path.dirname(a.out), os.path.basename(a.out))
        log(f"zip: {base}.zip ({os.path.getsize(base + '.zip') / 2**30:.2f} GB)")


if __name__ == "__main__":
    main()
