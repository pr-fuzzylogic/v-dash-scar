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
    Write-Host "Python not found in PATH. Install Python 3.10+ from https://www.python.org/downloads/ and re-run this script." -ForegroundColor Red
    exit 1
}
python --version

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
