#!/usr/bin/env bash
# install.sh - sets up an isolated Python virtual environment and installs
# all dependencies needed to run analyzer.py (Linux / macOS).
# V-DASH-SCAR - Vehicle Dashcam Scratch & Collision Automated Recognition
#
# Usage:
#   chmod +x install.sh
#   ./install.sh

set -e

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR=".venv"

echo "== Checking Python =="
if ! command -v "$PYTHON_BIN" &> /dev/null; then
    echo "Python3 not found. Install Python 3.10+ first: https://www.python.org/downloads/"
    exit 1
fi
"$PYTHON_BIN" --version

echo "== Checking ffmpeg =="
if ! command -v ffmpeg &> /dev/null; then
    echo "ffmpeg not found."
    if [[ "$(uname)" == "Darwin" ]]; then
        echo "Install with: brew install ffmpeg"
    else
        echo "Install with: sudo apt install ffmpeg   (Debian/Ubuntu)"
        echo "           or: sudo dnf install ffmpeg   (Fedora)"
    fi
    exit 1
fi
ffmpeg -version | head -n 1

echo "== Creating virtual environment in $VENV_DIR =="
"$PYTHON_BIN" -m venv "$VENV_DIR"

echo "== Activating virtual environment =="
source "$VENV_DIR/bin/activate"

echo "== Upgrading pip =="
pip install --upgrade pip

echo "== Installing Python dependencies =="
pip install -r requirements.txt

echo "== Downloading YOLO11 weights (nano detection + nano segmentation) =="
python - <<'PYEOF'
from ultralytics import YOLO
YOLO("yolo11n.pt")
YOLO("yolo11n-seg.pt")
print("YOLO weights downloaded.")
PYEOF

echo ""
echo "Setup complete. To use analyzer.py in a new terminal session run:"
echo "  source $VENV_DIR/bin/activate"
echo "  python v-dash-scar.py --sample sample.mp4 --input ./INPUT_DIR --output ./OUTPUT_DIR --methods 1,2,3"
