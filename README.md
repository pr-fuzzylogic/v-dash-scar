# Vehicle Dashcam Scratch & Collision Automated Recognition (V-DASH-SCAR)

![version](https://img.shields.io/badge/version-1.0.0-blue)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)

**Author:** [pr-fuzzylogic](https://github.com/pr-fuzzylogic)
**Repo:** https://github.com/pr-fuzzylogic/vehicle-dashcam-scratch-collision-automated-recognition

A tool for scanning large dashcam footage archives to locate a specific incident
(e.g. the moment your car got scratched) inside a user-defined region of interest
(ROI), without manually scrubbing through hundreds of hours of video.

---

## Table of contents

- [Why this exists](#why-this-exists)
- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation-recommended-isolated-venv-environment)
- [GPU acceleration (CUDA)](#gpu-acceleration-cuda)
- [Usage](#usage)
- [CLI parameters](#cli-parameters)
- [Output](#output)
- [Performance notes](#performance-notes)
- [Privacy and GDPR](#privacy-and-gdpr)
- [License](#license)

## Why this exists

This started from a real, annoying problem. A few times my car got scratched by
someone brushing past it, and my dashcam's built-in "collision/motion" sensor
found nothing - because the impact never actually moved or shook the car, so the
sensor never triggered an event recording. Meanwhile my regular continuous
footage clearly showed someone walking by and rubbing against the car. The
problem was that this footage added up to several hundred gigabytes, and
scrubbing through all of it by hand to find the ten seconds that mattered was not
realistic.

The idea behind this tool is simple: you take one sample recording from your
camera, mark a small region where you expect the culprit to have been (e.g. the
door panel, the bumper corner), then point the tool at the folder containing all
your footage and at an output folder for the results. You get back symlinks/links
to the original files that contain a hit, the extracted clips around each event,
and snapshot images from the moment of detection - so you can review just the
relevant seconds instead of hours of footage.

One practical tip: start by running the tool on just a handful of recordings
first, to see how your ROI selection behaves, before pointing it at your entire
archive. A poorly drawn region - for example one that includes a shiny part of
the hood - can easily produce false positives, since reflections of passing
objects (clouds, other cars, people on the sidewalk) moving across a glossy
surface can look like motion inside the ROI even though nothing actually touched
the car.

## How it works

A cascading pipeline with three methods, where each subsequent method only runs on
files that "passed" the previous one:

1. **Method 1 - Binary pixel difference** - fast, highly sensitive pixel difference
   inside the ROI between consecutive analyzed frames. Good as a first, coarse filter
   over a large batch of files.
2. **Method 2 - YOLO bounding boxes** - object detection (YOLO11n) with no class
   filtering. Detects anything that intersects the ROI (a person, a cart, a trash
   bin, a vehicle, etc.) - intentionally without a class filter, because the actual
   perpetrator may enter the frame after another object (e.g. a trash bin rolls in
   first, then a person appears).
3. **Method 3 - YOLO segmentation** - more accurate, slower segmentation
   (YOLO11n-seg) with class filtering (people, vehicles, animals) and mask-overlap
   checking (not just bounding box) against the ROI - for final, precise verification.
   Uses object tracking (ByteTrack) internally, which requires the `lapx` dependency.

If Method 1 is run first in the same session, Method 2 and/or Method 3 will
**only** scan the time windows flagged by Method 1's hits (with a time buffer),
drastically cutting analysis time. If Method 2 or 3 are run standalone (without
Method 1 having run earlier in that session), they scan the entire file from start
to end.

## Requirements

- Python 3.10 or newer
- ffmpeg available in PATH
- (optional, but recommended) GPU: NVIDIA CUDA or Apple Silicon (MPS) - the script
  automatically detects the available device and falls back to CPU if no GPU is found
- `lapx` - required by Method 3's ByteTrack object tracker (installed automatically
  via `requirements.txt`)

## Installation (recommended: isolated venv environment)

The installation creates a Python virtual environment (`.venv`) so it doesn't
interfere with your system-wide Python installation, and downloads the YOLO model
weights.

### Windows (PowerShell)

```powershell
git clone https://github.com/pr-fuzzylogic/vehicle-dashcam-scratch-collision-automated-recognition.git
cd vehicle-dashcam-scratch-collision-automated-recognition
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

If you prefer to do it manually, paste this into PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt'); YOLO('yolo11n-seg.pt')"
```

ffmpeg (if not already installed):

```powershell
winget install Gyan.FFmpeg
```

### Windows (cmd.exe)

```bat
git clone https://github.com/pr-fuzzylogic/vehicle-dashcam-scratch-collision-automated-recognition.git
cd vehicle-dashcam-scratch-collision-automated-recognition
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt'); YOLO('yolo11n-seg.pt')"
```

### macOS / Linux (bash/zsh)

```bash
git clone https://github.com/pr-fuzzylogic/vehicle-dashcam-scratch-collision-automated-recognition.git
cd vehicle-dashcam-scratch-collision-automated-recognition
chmod +x install.sh
./install.sh
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt'); YOLO('yolo11n-seg.pt')"
```

ffmpeg (if not already installed):

```bash
brew install ffmpeg        # macOS
sudo apt install ffmpeg    # Debian/Ubuntu
sudo dnf install ffmpeg    # Fedora
```

### Activating the environment in future sessions

After the initial installation, in a new terminal you only need to activate the
venv before using the script:

- Windows PowerShell: `.\.venv\Scripts\Activate.ps1`
- Windows cmd: `.venv\Scripts\activate.bat`
- macOS/Linux: `source .venv/bin/activate`

## GPU acceleration (CUDA)

`requirements.txt` installs the default `torch` build from PyPI, which on most
systems resolves to a **CPU-only** build (or a platform-default build that may not
include CUDA support). If you have an NVIDIA GPU and want hardware-accelerated
inference for Methods 2 and 3, install the CUDA-enabled build of PyTorch **instead
of** the plain `pip install torch` step, using PyTorch's official index:

```bash
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Replace `cu121` with the CUDA version matching your driver (e.g. `cu124`, `cu128`)
- check the exact command for your setup on the
[official PyTorch install selector](https://pytorch.org/get-started/locally/).

Apple Silicon (M1/M2/M3/M4) users get GPU acceleration automatically through MPS
with the standard `torch` package - no extra step needed.

To confirm which device the script will use, run:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('MPS available:', torch.backends.mps.is_available())"
```

`v-dash-scar.py` automatically picks the best available device in this order:
CUDA > MPS > CPU (see the console banner / `v-dash-scar.log` output at runtime for
confirmation of the device actually used).

## Usage

Running with `--version` prints the tool's name, version, and author without
launching the analysis:

```bash
python v-dash-scar.py --version
```

### Interactive mode (GUI)

```bash
python v-dash-scar.py
```

The script guides you through four steps via popup dialogs: select a sample video
file (used to draw the ROI with your mouse), select the input directory with
footage, select the output directory, and choose which methods to run.

> **Tip:** the first time you use the tool, or whenever you pick a new ROI, run it
> against a small subset of your archive first (a handful of files, not the whole
> few hundred GB) and check the `*_hit.jpg` results. This lets you catch a badly
> placed ROI early - for example a region that clips a reflective part of the
> bodywork - before you commit to a full, multi-hour scan of your entire archive.

### CLI mode

```bash
python v-dash-scar.py --sample file.mp4 --input ./FRONT --output ./Results --methods 1,2,3
```

Running only a single method (e.g. Method 2 alone, scanning entire files from
start to end without the range restriction from Method 1):

```bash
python v-dash-scar.py --sample file.mp4 --input ./FRONT --output ./Results --methods 2
```

## CLI parameters

| Parameter | Default | Description |
|---|---|---|
| `--version` | - | Prints name, version, and author, then exits |
| `--sample` | - | Path to the video file used to select the ROI |
| `--input` | - | Input directory with footage (searched recursively) |
| `--output` | - | Output directory for results |
| `--methods` | - | Comma-separated methods to run, e.g. `1,2,3` or `2` |
| `--extensions` | `.mp4,.mov,.avi,.ts` | Comma-separated video extensions to scan (case-insensitive, covers e.g. `.Mp4`, `.MP4`) |
| `--frame-step` | `5` | Analyze every Nth frame |
| `--pixel-diff-thresh` | `35` | Method 1: per-pixel grayscale difference threshold (0-255) |
| `--pixel-ratio-thresh` | `0.02` | Method 1: minimum fraction of changed pixels in the ROI (0.02 = 2%) |
| `--conf` | `0.30` | YOLO detection confidence threshold |
| `--merge-sec` | `10.0` | Merge hits into one event if the gap is below this many seconds |
| `--padding-sec` | `3.0` | Extra time margin added before/after each exported clip |
| `--fast-cut` | disabled | Fast clip extraction via `-c copy` (approximate, no re-encode) - default is an accurate re-encode |
| `--workers` | `2` | Number of parallel worker processes |
| `--resume` | disabled | Resume an interrupted run based on the saved `status.json` |
| `--verbose-timing` | disabled | Log detailed per-file decode/inference totals; without this flag, only a compact per-frame average is logged |

## Output

For each method, a subfolder is created (`Method_1_Binary`, `Method_2_YOLO_BBox`,
`Method_3_YOLO_Seg`) containing:

- `*_hit.jpg` - the first hit frame with the ROI (and detected objects) drawn on it
- `*_event_N.mp4` - an extracted clip for each detected event
- a link/copy of the original file (symlink, or hardlink/copy as a fallback if the OS doesn't support it)
- `report.csv` - a list of all processed files with status and event timestamps
- `status.json` - run state, used by `--resume`
- `v-dash-scar.log` - full run log, including a per-file timing line (`TIMING ...`
  by default, or `DIAGNOSTICS ...` with `--verbose-timing`) showing average
  decode and inference time per frame - useful for tuning `--workers` and
  `--frame-step` on large archives

## Performance notes

- The YOLO model is loaded once per worker process (not per file), which
  drastically reduces processing time when handling hundreds of files.
- ffmpeg clip extraction places `-ss` **before** `-i`, enabling a fast container-level
  seek instead of decoding the footage from second 0 - critical for long dashcam files.
- Restricting the scan range for Methods 2/3 to Method 1's time windows only applies
  if Method 1 actually ran earlier in the same session; running Method 2 or 3
  standalone scans the entire file.
- Automatic acceleration detection: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU.
- Each processed file logs its average decode time and average inference time per
  frame - if decode time dominates, the bottleneck is disk I/O / ffmpeg / OpenCV
  reading, not the model; if inference time dominates, consider a GPU or a smaller
  model.

## Privacy and GDPR

This tool analyzes footage that may contain third-party likenesses and license
plates. Before sharing extracted clips (e.g. for an insurance claim or a police
report), check local data protection regulations and consider limiting access to
authorized recipients only (police, insurer, property manager/HOA).

## License

MIT - see the `LICENSE` file.

---

*V-DASH-SCAR v1.0.0 - built by [pr-fuzzylogic](https://github.com/pr-fuzzylogic)*
