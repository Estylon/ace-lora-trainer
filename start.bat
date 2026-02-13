@echo off
REM ============================================================
REM ACE-Step LoRA Trainer — Quick Start (Windows)
REM ============================================================

REM Check if venv exists
if not exist "env\Scripts\activate.bat" (
    echo.
    echo [ERROR] Virtual environment not found!
    echo         Please run install.bat first.
    echo.
    pause
    exit /b 1
)

REM Activate and launch
call env\Scripts\activate.bat
python launch.py %*
