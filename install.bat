@echo off
REM ============================================================
REM ACE-Step LoRA Trainer + Captioner — Windows Installer
REM ============================================================
REM Creates a virtual environment and installs all dependencies.
REM Run this once before using the trainer.
REM ============================================================

echo.
echo ============================================================
echo   ACE-Step LoRA Trainer + Captioner - Installer
echo ============================================================
echo.

REM Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+ from https://python.org
    echo         Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

REM Show Python version
for /f "tokens=*" %%i in ('python --version 2^>^&1') do echo [INFO] Found %%i

REM Create virtual environment if it doesn't exist
if not exist "env\Scripts\activate.bat" (
    echo.
    echo [1/3] Creating virtual environment...
    python -m venv env
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo       Done.
) else (
    echo [1/3] Virtual environment already exists, skipping creation.
)

REM Activate the virtual environment
echo.
echo [2/3] Activating virtual environment...
call env\Scripts\activate.bat

REM Upgrade pip and install uv
echo.
echo [3/3] Installing dependencies (this may take several minutes)...
echo.
python -m pip install --upgrade pip >nul 2>&1
pip install uv >nul 2>&1

REM Install all requirements
uv pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [WARNING] uv install failed, falling back to pip...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo [ERROR] Installation failed. Check the errors above.
        pause
        exit /b 1
    )
)

REM Verify critical packages
echo.
echo ============================================================
echo   Verifying installation...
echo ============================================================
echo.

python -c "import torch; print(f'  [OK] PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')" 2>nul || echo   [FAIL] PyTorch not installed
python -c "import peft; print(f'  [OK] PEFT {peft.__version__}')" 2>nul || echo   [FAIL] PEFT not installed - LoRA training will NOT work!
python -c "import lightning; print(f'  [OK] Lightning {lightning.__version__}')" 2>nul || echo   [FAIL] Lightning not installed
python -c "import gradio; print(f'  [OK] Gradio {gradio.__version__}')" 2>nul || echo   [FAIL] Gradio not installed
python -c "import prodigyopt; print('  [OK] Prodigy optimizer')" 2>nul || echo   [FAIL] Prodigy optimizer not installed
python -c "import transformers; print(f'  [OK] Transformers {transformers.__version__}')" 2>nul || echo   [FAIL] Transformers not installed

echo.
echo ============================================================
echo   Installation complete!
echo ============================================================
echo.
echo   To start the trainer, run:  start.bat
echo   Or manually:  env\Scripts\activate ^&^& python launch.py
echo.
pause
