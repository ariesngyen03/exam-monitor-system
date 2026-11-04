@echo off
cd /d "%~dp0.."

echo ===================================================
echo Setting up Python Virtual Environment (venv)
echo ===================================================

if exist scripts\local_env.bat (
    call scripts\local_env.bat
)

if "%LOCAL_PYTHON_EXE%"=="" (
    set LOCAL_PYTHON_EXE=python
)

echo [0] Using Python at: %LOCAL_PYTHON_EXE%
"%LOCAL_PYTHON_EXE%" -c "import sys; sys.exit(0 if sys.version_info[:2] in ((3,10),(3,11)) else 1)"
if errorlevel 1 (
    echo [ERROR] AI can Python 3.10 hoac 3.11 ^(mediapipe.solutions khong co tren 3.12+^).
    echo         Dat LOCAL_PYTHON_EXE trong scripts\local_env.bat tro toi Python 3.11.
    pause
    exit /b 1
)
if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe -c "import sys; sys.exit(0 if sys.version_info[:2] in ((3,10),(3,11)) else 1)"
    if errorlevel 1 (
        echo [ERROR] venv hien tai khong phai Python 3.10/3.11. Xoa thu muc venv roi chay lai script nay.
        pause
        exit /b 1
    )
)

if not exist "venv\" (
    echo [1] Creating virtual environment...
    "%LOCAL_PYTHON_EXE%" -m venv venv
) else (
    echo [1] Virtual environment already exists.
)

echo [2] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3] Upgrading pip...
python -m pip install --upgrade pip

echo [4] Installing required AI libraries...
pip install -r python_ai\requirements.txt

echo ===================================================
echo Setup Complete! 
echo ===================================================
pause
