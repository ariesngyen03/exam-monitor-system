@echo off
:: Chay AI server va tu khoi dong lai neu AI bi dung giua chung. Start.bat tao data\.stop de ket thuc.
cd /d "%~dp0"
set "LOG=%~dp0data\logs\ai_console.log"
:loop
if exist "%~dp0data\.stop" goto :eof
echo ===== [%date% %time%] Khoi dong AI ===== >> "%LOG%"
"%~dp0python\python.exe" -u -s "%~dp0python_ai\main.py" --stop-file "%~dp0data\.stop" >> "%LOG%" 2>&1
if exist "%~dp0data\.stop" goto :eof
echo ===== [%date% %time%] AI dung (ma loi %errorlevel%), khoi dong lai sau 3 giay ===== >> "%LOG%"
timeout /t 3 /nobreak >nul
goto loop
