# install.ps1 - sets up an isolated Python virtual environment and installs
# all dependencies needed to run v-dash-scar.py (Windows PowerShell).
# V-DASH-SCAR - Vehicle Dashcam Scratch & Collision Automated Recognition
#
# Usage (PowerShell):
#   powershell -ExecutionPolicy Bypass -File .\install.ps1

$ErrorActionPreference = "Stop"

Write-Host "== Checking Python ==" -ForegroundColor Cyan
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "Python not found in PATH. Install Python 3.11 from https://www.python.org/downloads/ and re-run this script." -ForegroundColor Red
    exit 1
}
python --version

Write-Host "== Checking Tcl/Tk (needed for customtkinter / _tkinter) ==" -ForegroundColor Cyan
$tkCheck = python -c "import _tkinter" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "_tkinter is missing for this Python interpreter." -ForegroundColor Red
    Write-Host ""
    # The Windows Store build of Python (and some minimal/embeddable builds) ships without Tcl/Tk.
    # The official python.org installer always includes it, so the fix is almost always
    # "install the real python.org build" rather than adding a package.
    $pySource = python -c "import sys; print(sys.executable)"
    if ($pySource -like "*WindowsApps*") {
        Write-Host "This Python looks like it came from the Microsoft Store, which does NOT ship Tcl/Tk." -ForegroundColor Yellow
        Write-Host "Fix:"
        Write-Host "  1. Uninstall the Store version: Settings > Apps > Python 3.11 > Uninstall"
        Write-Host "  2. Install the official build from https://www.python.org/downloads/"
        Write-Host "     (during setup, keep the default 'tcl/tk and IDLE' feature checked)"
        Write-Host "  3. Re-run this script."
    } else {
        Write-Host "Fix:"
        Write-Host "  1. Re-run the Python installer from https://www.python.org/downloads/"
        Write-Host "     choose 'Modify' and make sure 'tcl/tk and IDLE' is checked, or do a fresh install."
        Write-Host "  2. Alternatively, if you use a Conda/Anaconda Python, run: conda install tk"
        Write-Host "  3. Re-run this script."
    }
    Write-Host ""
    Write-Host "If you already have a .venv folder, delete it after fixing Python so it gets rebuilt:" -ForegroundColor Yellow
    Write-Host "  Remove-Item -Recurse -Force .venv"
    exit 1
}
Write-Host "_tkinter OK."

Write-Host "== Checking ffmpeg ==" -ForegroundColor Cyan
$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ffmpeg) {
    Write-Host "ffmpeg not found in PATH." -ForegroundColor Red
    Write-Host "Install with winget:  winget install Gyan.FFmpeg"
    Write-Host "or download from:     https://www.gyan.dev/ffmpeg/builds/ and add the bin folder to PATH"
    exit 1
}
ffmpeg -version | Select-Object -First 1

Write-Host "== Creating virtual environment in .venv ==" -ForegroundColor Cyan
python -m venv .venv

Write-Host "== Activating virtual environment ==" -ForegroundColor Cyan
. .\.venv\Scripts\Activate.ps1

Write-Host "== Verifying tkinter is importable inside the venv ==" -ForegroundColor Cyan
python -c "import tkinter" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "tkinter is still not importable inside the venv." -ForegroundColor Red
    Write-Host "This usually means the base Python still lacks Tcl/Tk (see message above)." -ForegroundColor Yellow
    Write-Host "Fix Python first, then run: Remove-Item -Recurse -Force .venv ; .\install.ps1"
    exit 1
}

Write-Host "== Upgrading pip ==" -ForegroundColor Cyan
python -m pip install --upgrade pip

Write-Host "== Installing Python dependencies ==" -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "== Downloading YOLO11 weights (nano detection + nano segmentation) ==" -ForegroundColor Cyan
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt'); YOLO('yolo11n-seg.pt'); print('YOLO weights downloaded.')"

Write-Host ""
Write-Host "Setup complete. To use v-dash-scar.py in a new PowerShell session run:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python v-dash-scar.py --sample sample.mp4 --input .\INPUT_DIR --output .\OUTPUT_DIR --methods 1,2,3"
