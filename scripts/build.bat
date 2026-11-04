@echo off
cd /d "%~dp0.."

if exist scripts\local_env.bat (
    call scripts\local_env.bat
) else (
    echo [WARN] Khong tim thay scripts\local_env.bat, su dung duong dan mac dinh!
    set LOCAL_QT_DIR=C:\Qt\6.11.1\mingw_64
    set LOCAL_MSYS2_DIR=C:\msys64\mingw64
    set LOCAL_OPENCV_DIR=D:\CodeSpace\Work_Space\thesis_iuh\opencv_build
)

:: mingw32-make phai cung ho voi trinh bien dich mingw64. Ban cua MSYS2 ucrt64 lam ld loi
:: "ld returned 5 exit status" o buoc link -> mac dinh dung mingw32-make di kem Qt.
if "%LOCAL_MAKE%"=="" set LOCAL_MAKE=C:\Qt\Tools\mingw1310_64\bin\mingw32-make.exe

set PATH=%LOCAL_MSYS2_DIR%\bin;%LOCAL_QT_DIR%\bin;%PATH%
rmdir /S /Q build
mkdir build
cd build
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM="%LOCAL_MAKE%" -DCMAKE_PREFIX_PATH="%LOCAL_QT_DIR%" -DOpenCV_DIR="%LOCAL_OPENCV_DIR%" ..
cmake --build . -j 8
