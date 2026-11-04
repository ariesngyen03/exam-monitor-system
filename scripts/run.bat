@echo off
cd /d "%~dp0.."

if exist scripts\local_env.bat (
    call scripts\local_env.bat
) else (
    echo [WARN] Khong tim thay scripts\local_env.bat, su dung duong dan mac dinh!
    set LOCAL_QT_DIR=C:\Qt\6.11.1\mingw_64
    set LOCAL_MSYS2_DIR=C:\msys64\mingw64
    set LOCAL_OPENCV_DIR=D:\CodeSpace\Work_Space\thesis_iuh\opencv_build
    set LOCAL_ZMQ_DIR=D:\CodeSpace\Work_Space\thesis_iuh\zmq_workspace\zmq_install
)

:: Python chay AI: LOCAL_AI_PYTHON (neu co) > venv
set AI_PY=venv\Scripts\python.exe
if not "%LOCAL_AI_PYTHON%"=="" set AI_PY=%LOCAL_AI_PYTHON%

echo ===================================================
echo Starting Exam Monitor System - V1.1.0
echo ===================================================

echo [0] Cleaning up old processes...
:: Tat AI server cu (ke ca khi cua so da dong / chay an) de tranh trung cong 5556
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*python_ai*main.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >nul 2>&1
taskkill /F /T /FI "WINDOWTITLE eq AI Engine Python*" >nul 2>&1
taskkill /F /IM ExamMonitorSystem.exe >nul 2>&1

echo [1] Running Python AI Server (%AI_PY%)...
:: EXAM_ROOM / EXAM_PROCTOR / EXAM_CAMERA_INDEX trong local_env.bat duoc ke thua sang 2 cua so ben duoi
start "AI Engine Python" cmd /k ""%AI_PY%" python_ai\main.py"

echo [2] Waiting for AI models to load...
timeout /t 5 /nobreak >nul

echo [3] Running C++ GUI...
start "ExamMonitorSystem C++ GUI" cmd /k "set PATH=%LOCAL_QT_DIR%\bin;%LOCAL_MSYS2_DIR%\bin;%LOCAL_OPENCV_DIR%\bin;%LOCAL_ZMQ_DIR%\bin;%PATH% && .\build\ExamMonitorSystem.exe || echo C++ PROGRAM CRASHED! && pause"

echo ===================================================
echo Started! Bam BAT DAU tren giao dien; 15s dau AI hoc so do cho ngoi.
echo ===================================================
pause
