@echo off
:: ExamMonitorSystem - ban dong goi. Chay file nay de mo AI + giao dien. Xem HUONG_DAN.txt
cd /d "%~dp0"
title Exam Monitor System
set "ROOT=%~dp0"
set "EXAM_DATA_DIR=%ROOT%data"

:: --- Tuy chinh (chi co hieu luc lan dau; sau do dung trang Cai dat trong giao dien) ---
if not defined EXAM_CAMERA_INDEX set EXAM_CAMERA_INDEX=0
if not defined EXAM_ROOM set EXAM_ROOM=PHONG_THI_01
:: Do nhay AI: low (it bao sai nhat) | medium | high (bat nhieu nhat)
if not defined EXAM_SENSITIVITY set EXAM_SENSITIVITY=medium

:: Python trong goi: khong dung Python / thu vien da cai tren may nay
set PYTHONNOUSERSITE=1
set PYTHONIOENCODING=utf-8
set PYTHONPATH=
set PYTHONHOME=

if not exist "%ROOT%data\logs" mkdir "%ROOT%data\logs"
if exist "%ROOT%data\.stop" del "%ROOT%data\.stop"

echo [1/3] Tat phien cu (neu con)...
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*python_ai*main.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >nul 2>&1
taskkill /F /IM ExamMonitorSystem.exe >nul 2>&1

echo [2/3] Khoi dong AI (lan dau mat 20-60 giay, log: data\logs\ai_console.log)...
start "ExamMonitor AI" /min cmd /c ""%ROOT%ai_watchdog.bat""

echo [3/3] Mo giao dien. KHONG dong cua so nay - dong giao dien thi AI tu tat.
start "" /wait "%ROOT%app\ExamMonitorSystem.exe"

echo Dang luu bao cao va tat AI (toi da 20 giay)...
echo stop> "%ROOT%data\.stop"
:: AI thay file .stop -> dong phien, ghi bao cao roi tu thoat; qua 20 giay moi buoc tat
powershell -NoProfile -Command "$t=(Get-Date).AddSeconds(20); while ((Get-Date) -lt $t -and (Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*python_ai*main.py*' })) { Start-Sleep -Milliseconds 500 }; Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*python_ai*main.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >nul 2>&1
