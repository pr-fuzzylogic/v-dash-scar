#!/usr/bin/env bash
# install.sh - sets up an isolated Python virtual environment and installs
# all dependencies needed to run v-dash-scar.py (Linux / macOS).
# V-DASH-SCAR - Vehicle Dashcam Scratch & Collision Automated Recognition
#
# Usage:
#   chmod +x install.sh
#   ./install.sh

set -e

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR=".venv"

echo "== Checking Python =="
if ! command -v "$PYTHON_BIN" &> /dev/null; then
  echo "Python 3.11 not found. Install Python 3.11 first: https://www.python.org/downloads/"
  exit 1
fi
"$PYTHON_BIN" --version

# Extract the "3.11" style short version straight from the interpreter,
# so the Tk package we install always matches the Python we're using.
PY_SHORT_VER="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

echo "== Checking Tcl/Tk (needed for customtkinter / _tkinter) =="
if ! "$PYTHON_BIN" -c "import _tkinter" &> /dev/null; then
  echo "_tkinter is missing for $PYTHON_BIN (Python $PY_SHORT_VER)."
  if [[ "$(uname)" == "Darwin" ]]; then
    if ! command -v brew &> /dev/null; then
      echo "Homebrew not found. Install it from https://brew.sh, then run:"
      echo "  brew install python-tk@${PY_SHORT_VER}"
      exit 1
    fi
    echo "Installing python-tk@${PY_SHORT_VER} via Homebrew (Homebrew's python@${PY_SHORT_VER} ships without Tk by design)..."
    brew install "python-tk@${PY_SHORT_VER}" || {
      echo "Could not install python-tk@${PY_SHORT_VER}."
      echo "Check available versions with: brew search python-tk"
      exit 1
    }
  else
    echo "Attempting to install the OS Tk package..."
    if command -v apt &> /dev/null; then
      sudo apt update && sudo apt install -y "python${PY_SHORT_VER}-tk" || sudo apt install -y python3-tk
    elif command -v dnf &> /dev/null; then
      sudo dnf install -y python3-tkinter
    elif command -v pacman &> /dev/null; then
      sudo pacman -S --noconfirm tk
    else
      echo "Unsupported package manager. Install Tk manually (e.g. 'python3-tk') and re-run this script."
      exit 1
    fi
  fi

  # Re-check after attempting the fix.
  if ! "$PYTHON_BIN" -c "import _tkinter" &> /dev/null; then
    echo "_tkinter is still not importable with $PYTHON_BIN."
    echo "If you had a venv already, delete it (rm -rf $VENV_DIR) and re-run this script"
    echo "so the venv is rebuilt against the interpreter that now has Tk support."
    exit 1
  fi
  echo "Tk support confirmed for $PYTHON_BIN."
else
  echo "_tkinter OK."
fi

echo "== Checking ffmpeg =="
if ! command -v ffmpeg &> /dev/null; then
  echo "ffmpeg not found."
  if [[ "$(uname)" == "Darwin" ]]; then
    echo "Install with: brew install ffmpeg"
  else
    echo "Install with: sudo apt install ffmpeg (Debian/Ubuntu)"
    echo "         or: sudo dnf install ffmpeg (Fedora)"
  fi
  exit 1
fi
ffmpeg -version | head -n 1

echo "== Creating virtual environment in $VENV_DIR =="
"$PYTHON_BIN" -m venv "$VENV_DIR"

echo "== Activating virtual environment =="
source "$VENV_DIR/bin/activate"

echo "== Verifying tkinter is importable inside the venv =="
if ! python -c "import customtkinter" &> /dev/null; then
  if ! python -c "import tkinter" &> /dev/null; then
    echo "tkinter is still not importable inside the venv."
    echo "This usually means the venv was created before Tk support was installed."
    echo "Fix: rm -rf $VENV_DIR && ./install.sh"
    exit 1
  fi
fi

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
echo "Setup complete. To use v-dash-scar.py in a new terminal session run:"
echo "  source $VENV_DIR/bin/activate"
echo "  python v-dash-scar.py --sample sample.mp4 --input ./INPUT_DIR --output ./OUTPUT_DIR --methods 1,2,3"
