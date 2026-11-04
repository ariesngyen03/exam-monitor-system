@echo off
:: Sinh lai code Protobuf (C++ + Python) sau khi sua proto\exam_monitor.proto
:: protoc phai CUNG phien ban voi libprotobuf ma C++ link toi (MSYS2 mingw64).
cd /d "%~dp0.."

if exist scripts\local_env.bat call scripts\local_env.bat
if "%LOCAL_MSYS2_DIR%"=="" set LOCAL_MSYS2_DIR=C:\msys64\mingw64

"%LOCAL_MSYS2_DIR%\bin\protoc.exe" --version
"%LOCAL_MSYS2_DIR%\bin\protoc.exe" -Iproto --cpp_out=proto\cpp_out --python_out=proto\python_out proto\exam_monitor.proto
if errorlevel 1 (
    echo [ERROR] Sinh proto that bai.
    exit /b 1
)
if not "%LOCAL_AI_PYTHON%"=="" (
    "%LOCAL_AI_PYTHON%" -c "p=r'proto\python_out\exam_monitor_pb2.py'; s=open(p,encoding='utf-8').read(); s=s.replace('from google.protobuf import runtime_version as _runtime_version','try:\n    from google.protobuf import runtime_version as _runtime_version\nexcept ImportError:\n    _runtime_version = None').replace('_runtime_version.ValidateProtobufRuntimeVersion(','if _runtime_version is not None: _runtime_version.ValidateProtobufRuntimeVersion('); open(p,'w',encoding='utf-8').write(s)"
)
echo [OK] Da sinh proto\cpp_out va proto\python_out. Nho build lai C++ (scripts\build.bat).
