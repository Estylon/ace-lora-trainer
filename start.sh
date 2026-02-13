#!/bin/bash
# ============================================================
# ACE-Step LoRA Trainer — Quick Start (Linux/Mac)
# ============================================================

# Check if venv exists
if [ ! -f "env/bin/activate" ]; then
    echo ""
    echo "[ERROR] Virtual environment not found!"
    echo "        Please run install.sh first."
    echo ""
    exit 1
fi

# Activate and launch
source env/bin/activate
python launch.py "$@"
