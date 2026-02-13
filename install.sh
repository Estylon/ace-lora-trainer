#!/bin/bash
# ============================================================
# ACE-Step LoRA Trainer + Captioner — Linux/Mac Installer
# ============================================================
# Creates a virtual environment and installs all dependencies.
# Run this once before using the trainer.
# ============================================================

set -e

echo ""
echo "============================================================"
echo "  ACE-Step LoRA Trainer + Captioner - Installer"
echo "============================================================"
echo ""

# Check Python is available
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 not found. Please install Python 3.10+."
    echo "        Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip"
    echo "        macOS: brew install python3"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1)
echo "[INFO] Found $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -f "env/bin/activate" ]; then
    echo ""
    echo "[1/3] Creating virtual environment..."
    python3 -m venv env
    echo "      Done."
else
    echo "[1/3] Virtual environment already exists, skipping creation."
fi

# Activate the virtual environment
echo ""
echo "[2/3] Activating virtual environment..."
source env/bin/activate

# Upgrade pip and install uv
echo ""
echo "[3/3] Installing dependencies (this may take several minutes)..."
echo ""
python -m pip install --upgrade pip > /dev/null 2>&1
pip install uv > /dev/null 2>&1

# Install all requirements
if ! uv pip install -r requirements.txt; then
    echo ""
    echo "[WARNING] uv install failed, falling back to pip..."
    pip install -r requirements.txt
fi

# Verify critical packages
echo ""
echo "============================================================"
echo "  Verifying installation..."
echo "============================================================"
echo ""

python -c "import torch; print(f'  [OK] PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')" 2>/dev/null || echo "  [FAIL] PyTorch not installed"
python -c "import peft; print(f'  [OK] PEFT {peft.__version__}')" 2>/dev/null || echo "  [FAIL] PEFT not installed - LoRA training will NOT work!"
python -c "import lightning; print(f'  [OK] Lightning {lightning.__version__}')" 2>/dev/null || echo "  [FAIL] Lightning not installed"
python -c "import gradio; print(f'  [OK] Gradio {gradio.__version__}')" 2>/dev/null || echo "  [FAIL] Gradio not installed"
python -c "import prodigyopt; print('  [OK] Prodigy optimizer')" 2>/dev/null || echo "  [FAIL] Prodigy optimizer not installed"
python -c "import transformers; print(f'  [OK] Transformers {transformers.__version__}')" 2>/dev/null || echo "  [FAIL] Transformers not installed"

echo ""
echo "============================================================"
echo "  Installation complete!"
echo "============================================================"
echo ""
echo "  To start the trainer, run:  ./start.sh"
echo "  Or manually:  source env/bin/activate && python launch.py"
echo ""
