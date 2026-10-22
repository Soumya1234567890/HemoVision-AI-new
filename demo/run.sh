#!/usr/bin/env bash
# HemoVision AI Linux / macOS 1-Click Launcher
set -e

echo "======================================================================"
echo "  HemoVision AI: IDA & Beta Thalassemia Trait Decision Support"
echo "======================================================================"

if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 could not be found. Please install Python 3.10+."
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "[*] Creating virtual environment (.venv)..."
    python3 -m venv .venv
fi

echo "[*] Activating virtual environment..."
source .venv/bin/activate

echo "[*] Installing / verifying dependencies..."
pip install -r requirements.txt --quiet

echo "[*] Launching HemoVision AI..."
python3 main.py
