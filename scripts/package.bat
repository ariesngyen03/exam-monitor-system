@echo off
:: Dong goi ban chay duoc tren may khac -> dist\ExamMonitorSystem (them --zip de nen). Can build truoc.
cd /d "%~dp0.."
if exist scripts\local_env.bat call scripts\local_env.bat
set PKG_PY=python
if not "%LOCAL_AI_PYTHON%"=="" set PKG_PY=%LOCAL_AI_PYTHON%
"%PKG_PY%" scripts\package.py %*
pause
