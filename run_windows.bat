@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\pythonw.exe" (
    echo Hand-Bridge setup is required.
    echo Run setup_windows.ps1 first.
    pause
    exit /b 1
)

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "TF_CPP_MIN_LOG_LEVEL=2"
start "Hand-Bridge" /B ".venv\Scripts\pythonw.exe" "gui.py"
