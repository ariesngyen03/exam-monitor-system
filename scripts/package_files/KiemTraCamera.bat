@echo off
:: Kiem tra camera: dinh dang (MJPG/YUY2) va so frame/giay thuc te. Dong giao dien ExamMonitor truoc khi chay.
:: Ket qua ghi them vao data\logs\camera_check.log. Doi so camera: KiemTraCamera.bat 1
cd /d "%~dp0"
set PYTHONNOUSERSITE=1
set PYTHONIOENCODING=utf-8
set PYTHONPATH=
set PYTHONHOME=
set "CAM=%~1"
if "%CAM%"=="" set CAM=0
if not exist "%~dp0data\logs" mkdir "%~dp0data\logs"
echo ===== [%date% %time%] Camera %CAM% ===== >> "%~dp0data\logs\camera_check.log"
"%~dp0python\python.exe" -s "%~dp0python_ai\tools\camera_check.py" --index %CAM% > "%~dp0data\logs\camera_check.tmp" 2>&1
type "%~dp0data\logs\camera_check.tmp"
type "%~dp0data\logs\camera_check.tmp" >> "%~dp0data\logs\camera_check.log"
del "%~dp0data\logs\camera_check.tmp"
pause
