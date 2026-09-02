#!/usr/bin/env python3
"""
============================================================================
Vehicle Dashcam Scratch & Collision Automated Recognition (V-DASH-SCAR)
============================================================================

Author: pr-fuzzylogic
Repo: https://github.com/pr-fuzzylogic/vehicle-dashcam-scratch-collision-automated-recognition
Version: 1.1.0
License: MIT (see LICENSE)

Description:
Cascade pipeline for triaging large dashcam footage archives to locate
a specific incident (e.g. a scratch, bump, or collision) inside a
user-defined region of interest (ROI), without manually scrubbing
through hours/terabytes of video.

Methods (run individually or chained; each stage filters the file list
passed to the next):
1. Binary pixel difference - fast, high-sensitivity motion filter
2. YOLO bounding boxes - object detection, no class filtering
3. YOLO segmentation - class-filtered, precise mask overlap

Output (per method, under /Method_X_.../):
    *_hit.jpg        first frame where a hit was detected (ROI + boxes drawn)
    *_event_N.mp4    extracted clip for each detected event
    symlink/hardlink/copy to the source video
    report.csv       per-file status (hit/clean/error) and event timestamps
    status.json      run state, used for --resume
    v-dash-scar.log  full run log

Usage example:
    python v-dash-scar.py --sample sample.mp4 --input ./INPUT_DIR --output ./OUTPUT_DIR --methods 1,2,3

CHANGELOG (1.0.0 -> 1.1.0):
- FIX: check_mask_intersection() only tested whether a mask *vertex* fell
  inside the ROI, which produced false negatives whenever the ROI sat
  entirely inside a larger mask (no vertex ever lands inside it) or a mask
  edge crossed the ROI without a vertex inside it. Replaced with a
  rasterized overlap test (fillPoly + bitwise AND against the ROI region),
  which is a true polygon/rectangle intersection test.
- FIX: Method 3 reused a single YOLO model instance (with persist=True
  ByteTrack state) across every video handled by a given worker process.
  Track state now leaked between unrelated files. The tracker/predictor is
  now reset at the start of every new video.
- FIX: GUI widgets (progress bar, log textbox) were being mutated directly
  from the background analysis thread. Tkinter/customtkinter is not
  thread-safe; all cross-thread UI updates are now marshaled through
  self.after(0, ...).
- FIX: --resume was accepted by argparse and documented (status.json) but
  never implemented. status.json is now written after each method finishes
  and, with --resume, is loaded to skip already-processed files per method.
- FIX: --methods was parsed but silently ignored; the GUI always ran all
  three methods regardless of the flag. The checkboxes are now
  pre-populated from --methods when provided.
- FIX: cv2.CAP_PROP_POS_FRAMES seeks are only approximate on long-GOP
  footage (they land on the nearest keyframe, not the requested frame).
  Window-restricted scanning now reads back the actual landed position via
  cap.get(CAP_PROP_POS_FRAMES) after seeking and uses that as the true
  frame_idx baseline instead of trusting the requested start_frame,
  preventing systematic timestamp drift on long dashcam files.
============================================================================
"""

import os
import sys
import json
import csv
import shutil
import logging
import argparse
import subprocess
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import customtkinter as ctk
from PIL import Image

import cv2
import numpy as np
import torch
from ultralytics import YOLO

__title__ = "Vehicle Dashcam Scratch & Collision Automated Recognition"
__short_name__ = "V-DASH-SCAR"
__author__ = "pr-fuzzylogic"
__version__ = "1.1.0"
__license__ = "MIT"
__repo__ = "https://github.com/pr-fuzzylogic/vehicle-dashcam-scratch-collision-automated-recognition"

DEFAULT_EXTENSIONS = [".mp4", ".mov", ".avi", ".ts"]

# Extra frames appended to the end of a Method-1-derived window when
# restricting Methods 2/3 scanning range, to protect against rounding
# errors when converting seconds -> frame indices (avoids clipping the
# real event by one frame too early).
WINDOW_END_FRAME_BUFFER = 15

# How many frames before the requested start_frame we seek to first, so we
# can detect exactly where the decoder actually landed (long-GOP footage
# rarely seeks to the exact requested frame) and correct our frame_idx
# bookkeeping accordingly instead of trusting the request blindly.
SEEK_SAFETY_MARGIN_FRAMES = 5

STATUS_FILENAME = "status.json"

# COCO class ids used by Method 3's class filter:
# 0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck,
# 24 backpack, 26 handbag, 28 suitcase, 36 skateboard
METHOD3_CLASSES = [0, 1, 2, 3, 5, 7, 24, 26, 28, 36]


# ============================================================================
# Banner and version info
# ============================================================================

def print_banner():
    banner = f"""
+----------------------------------------------------------------------+
{__short_name__} - {__title__}
Version {__version__} | Author: {__author__} | License: {__license__}
{__repo__}
+----------------------------------------------------------------------+
Run interactively: ./v-dash-scar.py
Run via CLI:       ./v-dash-scar.py --help
+----------------------------------------------------------------------+
"""
    print(banner, flush=True)


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        prog=__short_name__,
        description=f"{__title__} ({__short_name__}) v{__version__} - dashcam footage triage tool (motion diff + YOLO cascade)"
    )

    parser.add_argument("--version", action="store_true", help="Print version info and exit")
    parser.add_argument("--sample", type=str, help="Path to sample video file (used to pick ROI)")
    parser.add_argument("--input", type=str, help="Input directory with video files")
    parser.add_argument("--output", type=str, help="Output directory for results")
    parser.add_argument("--methods", type=str, help="Comma separated methods to run in order, e.g. 1,2,3")
    parser.add_argument("--extensions", type=str, default=",".join(DEFAULT_EXTENSIONS),
                         help="Comma separated video extensions to scan, e.g. .mp4,.mov,.avi (case-insensitive)")
    parser.add_argument("--frame-step", type=int, default=5,
                         help="Analyze every Nth frame (default 5)")
    parser.add_argument("--pixel-diff-thresh", type=int, default=35,
                         help="Method 1: per-pixel grayscale difference threshold (default 35)")
    parser.add_argument("--pixel-ratio-thresh", type=float, default=0.02,
                         help="Method 1: fraction of ROI pixels that must change to trigger a hit (default 0.02 = 2%%)")
    parser.add_argument("--conf", type=float, default=0.30, help="YOLO confidence threshold (default 0.30)")
    parser.add_argument("--merge-sec", type=float, default=10.0,
                         help="Merge hits into one event if gap is below this many seconds (default 10)")
    parser.add_argument("--padding-sec", type=float, default=3.0,
                         help="Extra seconds of padding added before/after each exported clip (default 3)")
    parser.add_argument("--fast-cut", action="store_true",
                         help="Use fast keyframe-based ffmpeg cut (-c copy). Less precise, no re-encode.")
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel worker processes (default 2)")
    parser.add_argument("--bump-detection", action="store_true", help="Enable global camera bump detection filtering")
    parser.add_argument("--resume", action="store_true",
                         help="Skip files already recorded in a previous run's status.json for the same method")
    parser.add_argument("--verbose-timing", action="store_true",
                         help="Log per-file decode/inference timing diagnostics (default: only a per-frame average summary is logged)")
    return parser.parse_args()


# ============================================================================
# Geometry helpers
# ============================================================================

def check_bbox_intersection(box, roi):
    bx1, by1, bx2, by2 = box
    rx1, ry1, rx2, ry2 = roi
    return not (bx2 < rx1 or bx1 > rx2 or by2 < ry1 or by1 > ry2)


def check_mask_intersection(mask_points, roi, frame_shape):
    """True polygon/rectangle intersection test.

    The previous implementation only checked whether a mask *vertex* fell
    inside the ROI rectangle. That produces false negatives whenever the ROI
    sits entirely inside a larger mask (no vertex is ever inside it) or a
    mask edge crosses the ROI without any vertex landing inside it. This
    version rasterizes the mask polygon and the ROI onto small boolean
    canvases and checks for a real pixel-level overlap, which is a correct
    (if slightly more expensive) test.
    """
    if mask_points is None or len(mask_points) == 0:
        return False

    rx1, ry1, rx2, ry2 = [int(v) for v in roi]
    h, w = frame_shape[:2]
    rx1, ry1 = max(0, rx1), max(0, ry1)
    rx2, ry2 = min(w, rx2), min(h, ry2)
    if rx2 <= rx1 or ry2 <= ry1:
        return False

    mask_canvas = np.zeros((h, w), dtype=np.uint8)
    poly = np.array(mask_points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask_canvas, [poly], 1)

    roi_slice = mask_canvas[ry1:ry2, rx1:rx2]
    return bool(roi_slice.any())


def merge_windows(windows, padding_sec):
    """Expand each (start,end) window by padding_sec and merge overlaps."""
    if not windows:
        return []
    padded = sorted((max(0.0, s - padding_sec), e + padding_sec) for s, e in windows)
    merged = [list(padded[0])]
    for s, e in padded[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(w) for w in merged]


# ============================================================================
# Case-insensitive recursive file discovery (handles ".Mp4" style names that
# a plain .lower()/.upper() glob pattern would silently miss).
# ============================================================================

def find_video_files(input_dir, extensions):
    wanted = {e.lower() for e in extensions}
    matches = []
    for root, _dirs, files in os.walk(input_dir):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in wanted:
                matches.append(os.path.join(root, name))
    return sorted(set(matches))


# ============================================================================
# status.json handling (--resume)
# ============================================================================

def load_status(output_dir):
    path = os.path.join(output_dir, STATUS_FILENAME)
    if not os.path.exists(path):
        return {"methods": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logging.warning(f"Could not read {path}, starting fresh: {e}")
        return {"methods": {}}


def save_status(output_dir, status):
    path = os.path.join(output_dir, STATUS_FILENAME)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
    os.replace(tmp_path, path)


# ============================================================================
# Device / model handling (loaded ONCE per worker process, not per file)
# ============================================================================

_WORKER_MODEL = None
_WORKER_DEVICE = None


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def init_worker(method):
    """Runs once per worker process when the pool for a given method is created."""
    global _WORKER_MODEL, _WORKER_DEVICE
    _WORKER_DEVICE = get_device()
    if method == 2:
        _WORKER_MODEL = YOLO("yolo11n.pt")
    elif method == 3:
        _WORKER_MODEL = YOLO("yolo11n-seg.pt")
    else:
        _WORKER_MODEL = None
    logging.info(f"Worker ready | method={method} device={_WORKER_DEVICE} pid={os.getpid()}")


def reset_tracker_state(model):
    """Force ByteTrack/predictor state to be reinitialized on the next
    .track() call. Without this, a model instance reused across multiple
    video files within the same worker process (persist=True) carries
    track history from the *previous* file into the next one."""
    if model is not None:
        model.predictor = None


# ============================================================================
# ffmpeg clip extraction
# ============================================================================

def extract_clip(video_path, start_sec, end_sec, padding, output_path, fast_cut=False):
    """-ss is placed BEFORE -i in both branches. When -ss precedes -i, ffmpeg
    seeks directly to the nearest keyframe at the container/demuxer level
    instead of decoding the stream from second 0, which is critical for
    long dashcam files (otherwise every cut on a 1-hour file would decode
    the whole hour first)."""
    clip_start = max(0.0, start_sec - padding)
    clip_end = end_sec + padding
    duration = clip_end - clip_start

    if fast_cut:
        cmd = ["ffmpeg", "-y", "-nostdin",
               "-ss", str(clip_start), "-i", video_path,
               "-t", str(duration), "-c", "copy", output_path]
    else:
        # Re-encode for frame-accurate cuts, important for evidence purposes.
        # -ss before -i: fast input seek, then a short accurate decode of
        # only the `duration` window, not the whole file.
        cmd = ["ffmpeg", "-y", "-nostdin",
               "-ss", str(clip_start), "-i", video_path, "-t", str(duration),
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
               "-c:a", "aac", output_path]

    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        logging.warning(f"ffmpeg failed for {output_path}: {result.stderr.decode(errors='ignore')[:300]}")
        return False
    return True


# ============================================================================
# Frame iteration (with optional restriction to time windows from Method 1)
# ============================================================================

def iter_frames(cap, fps, frame_step, windows=None):
    """Yield (frame_idx, frame). If windows is given, seek directly to each
    window instead of decoding the whole file, drastically cutting time for
    methods 2/3 when Method 1 already narrowed down candidate segments.

    cv2.CAP_PROP_POS_FRAMES seeks are only approximate on long-GOP footage:
    the decoder typically lands on the nearest keyframe, not the exact
    requested frame. To avoid silent timestamp drift, we seek a small
    safety margin *before* the requested start, then read back the actual
    landed position via cap.get(CAP_PROP_POS_FRAMES) and use that as the
    true frame_idx baseline instead of trusting the request blindly.
    """
    if not windows:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                return
            if frame_idx % frame_step == 0:
                yield frame_idx, frame
            frame_idx += 1
    else:
        for w_start, w_end in windows:
            requested_start = max(0, int(w_start * fps))
            seek_target = max(0, requested_start - SEEK_SAFETY_MARGIN_FRAMES)
            end_frame = int(w_end * fps) + 1 + WINDOW_END_FRAME_BUFFER

            cap.set(cv2.CAP_PROP_POS_FRAMES, seek_target)
            actual_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            frame_idx = actual_pos if actual_pos >= 0 else seek_target

            while frame_idx <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx >= requested_start - SEEK_SAFETY_MARGIN_FRAMES and frame_idx % frame_step == 0:
                    yield frame_idx, frame
                frame_idx += 1


# ============================================================================
# Safe "link" to original file (symlink -> hardlink -> copy fallback)
# ============================================================================

def link_or_copy(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.symlink(src, dst)
        return
    except FileExistsError:
        return
    except OSError:
        pass
    try:
        os.link(src, dst)
        return
    except OSError:
        pass
    try:
        shutil.copy2(src, dst)
    except OSError as e:
        logging.warning(f"Could not link or copy {src} -> {dst}: {e}")


# ============================================================================
# Core per-video worker task
# ============================================================================

def process_video_task(task):
    (video_path, output_dir, roi_coords, method, restrict_windows, opts) = task

    base_name = os.path.basename(video_path)
    rx1, ry1, rx2, ry2 = roi_coords
    roi_area = max(1, (rx2 - rx1) * (ry2 - ry1))

    try:
        # Request hardware-accelerated decoding across platforms
        params = [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY]
        cap = cv2.VideoCapture(video_path, cv2.CAP_ANY, params)
        if not cap.isOpened():
            return video_path, False, [], "Could not open file"

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        frame_step = opts["frame_step"]
        merge_threshold_sec = opts["merge_sec"]
        padding_sec = opts["padding_sec"]
        verbose_timing = opts.get("verbose_timing", False)
        bump_detection = opts.get("bump_detection", False)

        windows = merge_windows(restrict_windows, padding_sec) if restrict_windows else None

        events = []
        current_event_start = None
        current_event_end = None
        first_hit_frame_saved = False
        hit_found_in_file = False
        prev_roi_gray = None
        prev_global_gray = None

        model = _WORKER_MODEL
        device = _WORKER_DEVICE

        # Method 3 uses ByteTrack with persist=True. The model instance is
        # shared across every video handled by this worker process, so
        # tracker state from a *previous* file would otherwise leak into
        # this one. Force a clean tracker per video.
        if method == 3:
            reset_tracker_state(model)

        total_decode_time = 0.0
        total_inference_time = 0.0
        frames_processed = 0

        decode_start = time.perf_counter()

        for frame_idx, frame in iter_frames(cap, fps, frame_step, windows):
            decode_end = time.perf_counter()
            total_decode_time += (decode_end - decode_start)
            frames_processed += 1

            hit = False
            debug_frame = None
            boxes = None
            gray_frame_cache = None

            inference_start = time.perf_counter()

            bump_detected = False
            if bump_detection:
                gray_frame_cache = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                small_gray = cv2.resize(gray_frame_cache, (320, 180))
                curr_global_gray = np.float32(small_gray)

                if prev_global_gray is not None:
                    shift, response = cv2.phaseCorrelate(prev_global_gray, curr_global_gray)
                    mag = np.sqrt(shift[0]**2 + shift[1]**2)
                    if mag > 2.0:
                        bump_detected = True
                prev_global_gray = curr_global_gray

            if bump_detected:
                inference_end = time.perf_counter()
                total_inference_time += (inference_end - inference_start)
                decode_start = time.perf_counter()
                continue

            if method == 1:
                if gray_frame_cache is None:
                    gray_frame_cache = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                curr_roi_gray = gray_frame_cache[ry1:ry2, rx1:rx2]
                if curr_roi_gray.size > 0:
                    curr_roi_gray = cv2.GaussianBlur(curr_roi_gray, (7, 7), 0)

                    if prev_roi_gray is not None and frame_idx > 30:
                        diff = cv2.absdiff(curr_roi_gray, prev_roi_gray)
                        changed_pixels = (diff > opts["pixel_diff_thresh"]).sum()
                        if (changed_pixels / roi_area) > opts["pixel_ratio_thresh"]:
                            hit = True
                            debug_frame = frame.copy()
                            cv2.rectangle(debug_frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
                    prev_roi_gray = curr_roi_gray

            elif method == 2:
                results = model.predict(source=frame, conf=opts["conf"], imgsz=640,
                                         device=device, verbose=False)
                if results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    for box in boxes:
                        if check_bbox_intersection(box, roi_coords):
                            hit = True
                            break
                    if hit:
                        debug_frame = frame.copy()
                        cv2.rectangle(debug_frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
                        for box in boxes:
                            bx1, by1, bx2, by2 = map(int, box)
                            color = (0, 255, 0) if check_bbox_intersection(box, roi_coords) else (255, 0, 0)
                            cv2.rectangle(debug_frame, (bx1, by1), (bx2, by2), color, 2)

            elif method == 3:
                results = model.track(source=frame, conf=opts["conf"], imgsz=640,
                                       classes=METHOD3_CLASSES,
                                       persist=True, tracker="bytetrack.yaml",
                                       device=device, verbose=False)
                if results[0].boxes is not None and results[0].masks is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    masks = results[0].masks.xy
                    for box, mask in zip(boxes, masks):
                        if check_bbox_intersection(box, roi_coords) and check_mask_intersection(mask, roi_coords, frame.shape):
                            hit = True
                            break
                    if hit:
                        debug_frame = frame.copy()
                        cv2.rectangle(debug_frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
                        for box in boxes:
                            bx1, by1, bx2, by2 = map(int, box)
                            cv2.rectangle(debug_frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

            inference_end = time.perf_counter()
            total_inference_time += (inference_end - inference_start)

            if hit:
                hit_found_in_file = True
                current_sec = frame_idx / fps

                if not first_hit_frame_saved and debug_frame is not None:
                    out_jpg = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}_hit.jpg")
                    cv2.imwrite(out_jpg, debug_frame)
                    first_hit_frame_saved = True

                if current_event_start is None:
                    current_event_start = current_sec
                    current_event_end = current_sec
                elif current_sec <= current_event_end + merge_threshold_sec:
                    current_event_end = current_sec
                else:
                    events.append((current_event_start, current_event_end))
                    current_event_start = current_sec
                    current_event_end = current_sec

            decode_start = time.perf_counter()

        if current_event_start is not None:
            events.append((current_event_start, current_event_end))

        cap.release()

        if frames_processed > 0:
            avg_decode_ms = (total_decode_time / frames_processed) * 1000
            avg_inference_ms = (total_inference_time / frames_processed) * 1000
            if verbose_timing:
                logging.info(
                    f"DIAGNOSTICS {base_name} | frames={frames_processed} "
                    f"decode_total={total_decode_time:.2f}s inference_total={total_inference_time:.2f}s "
                    f"avg_decode={avg_decode_ms:.1f}ms/frame avg_inference={avg_inference_ms:.1f}ms/frame"
                )
            else:
                logging.info(
                    f"TIMING {base_name} | frames={frames_processed} "
                    f"avg_decode={avg_decode_ms:.1f}ms/frame avg_inference={avg_inference_ms:.1f}ms/frame"
                )

        if hit_found_in_file:
            link_or_copy(video_path, os.path.join(output_dir, base_name))
            for idx, (start_sec, end_sec) in enumerate(events):
                out_clip = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}_event_{idx}.mp4")
                extract_clip(video_path, start_sec, end_sec, padding_sec, out_clip,
                             fast_cut=opts["fast_cut"])

        return video_path, hit_found_in_file, events, None

    except Exception as e:
        logging.exception(f"Error processing {video_path}")
        return video_path, False, [], str(e)


def write_unified_report(output_dir, unified_db):
    csv_path = os.path.join(output_dir, "report.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "m1_hit", "m2_hit", "m3_hit", "final_start", "final_end", "error"])
        for vid_path, data in unified_db.items():
            writer.writerow([
                vid_path,
                data.get(1, {}).get("hit", False),
                data.get(2, {}).get("hit", False),
                data.get(3, {}).get("hit", False),
                data.get("start", ""),
                data.get("end", ""),
                data.get("error", "")
            ])


class LogRedirector:
    def __init__(self, app):
        self.app = app

    def write(self, message):
        if message.strip():
            # Marshal into the main thread: this may be called from the
            # background analysis thread, and Tkinter/customtkinter widgets
            # must only be touched from the main thread.
            self.app.after(0, self.app.log, message)

    def flush(self):
        pass


class VDashScarApp(ctk.CTk):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.title(f"{__short_name__} v{__version__}")
        self.geometry("1000x700")

        self.sample_path = ctk.StringVar(value=args.sample if args.sample else "")
        self.input_dir = ctk.StringVar(value=args.input if args.input else "")
        self.output_dir = ctk.StringVar(value=args.output if args.output else "")

        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)

        self.tab_config = self.tabview.add("Configuration")
        self.tab_progress = self.tabview.add("Progress")
        self.tab_review = self.tabview.add("Incident Browser")

        # --methods was previously parsed but ignored; the GUI checkboxes
        # always defaulted to all three methods enabled regardless of the
        # flag. Pre-populate them from --methods when it was supplied.
        requested_methods = {1, 2, 3}
        if args.methods:
            try:
                requested_methods = {int(m.strip()) for m in args.methods.split(",") if m.strip()}
            except ValueError:
                logging.warning(f"Could not parse --methods={args.methods!r}, defaulting to 1,2,3")
                requested_methods = {1, 2, 3}

        self.build_config_tab(requested_methods)
        self.build_progress_tab()
        self.build_review_tab()

        self.unified_db = {}
        self.video_cap = None
        self.playing = False
        self.play_after_id = None
        self.video_fps = 30.0
        self.playback_speed = 1.0
        self.roi_coords = None

    def build_config_tab(self, requested_methods):
        frame = ctk.CTkFrame(self.tab_config)
        frame.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(frame, text="Sample Video Path").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        ctk.CTkEntry(frame, textvariable=self.sample_path, width=400).grid(row=0, column=1, padx=10, pady=10)
        ctk.CTkButton(frame, text="Browse", command=self.browse_sample).grid(row=0, column=2, padx=10, pady=10)

        ctk.CTkLabel(frame, text="Input Directory").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        ctk.CTkEntry(frame, textvariable=self.input_dir, width=400).grid(row=1, column=1, padx=10, pady=10)
        ctk.CTkButton(frame, text="Browse", command=self.browse_input).grid(row=1, column=2, padx=10, pady=10)

        ctk.CTkLabel(frame, text="Output Directory").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        ctk.CTkEntry(frame, textvariable=self.output_dir, width=400).grid(row=2, column=1, padx=10, pady=10)
        ctk.CTkButton(frame, text="Browse", command=self.browse_output).grid(row=2, column=2, padx=10, pady=10)

        self.m1_var = ctk.BooleanVar(value=1 in requested_methods)
        self.m2_var = ctk.BooleanVar(value=2 in requested_methods)
        self.m3_var = ctk.BooleanVar(value=3 in requested_methods)

        ctk.CTkCheckBox(frame, text="Method 1 Binary pixel difference", variable=self.m1_var).grid(row=3, column=0, columnspan=3, padx=10, pady=5, sticky="w")
        ctk.CTkCheckBox(frame, text="Method 2 YOLO bounding boxes", variable=self.m2_var).grid(row=4, column=0, columnspan=3, padx=10, pady=5, sticky="w")
        ctk.CTkCheckBox(frame, text="Method 3 YOLO segmentation", variable=self.m3_var).grid(row=5, column=0, columnspan=3, padx=10, pady=5, sticky="w")

        self.resume_var = ctk.BooleanVar(value=bool(self.args.resume))
        ctk.CTkCheckBox(frame, text="Resume (skip files already in status.json)", variable=self.resume_var).grid(row=6, column=0, columnspan=3, padx=10, pady=5, sticky="w")

        self.bump_var = ctk.BooleanVar(value=bool(self.args.bump_detection))
        ctk.CTkCheckBox(frame, text="Enable Camera Bump Detection", variable=self.bump_var).grid(row=7, column=0, columnspan=3, padx=10, pady=5, sticky="w")

        ctk.CTkButton(frame, text="Start Analysis", command=self.start_processing, fg_color="green").grid(row=8, column=0, columnspan=3, pady=30)

    def build_progress_tab(self):
        self.log_box = ctk.CTkTextbox(self.tab_progress, state="normal", wrap="word")
        self.log_box.pack(fill="both", expand=True, padx=10, pady=10)

        self.progress_bar = ctk.CTkProgressBar(self.tab_progress)
        self.progress_bar.pack(fill="x", padx=10, pady=10)
        self.progress_bar.set(0)

        self.stop_button = ctk.CTkButton(self.tab_progress, text="Stop Analysis", command=self.stop_processing, fg_color="red", state="disabled")
        self.stop_button.pack(pady=5)

        self.is_cancelled = False

    def stop_processing(self):
        self.is_cancelled = True
        self.log("Stopping analysis... waiting for active tasks to finish.")
        self.stop_button.configure(state="disabled")

    def build_review_tab(self):
        self.review_split = ctk.CTkFrame(self.tab_review)
        self.review_split.pack(fill="both", expand=True)

        self.clip_listbox = ctk.CTkScrollableFrame(self.review_split, width=250)
        self.clip_listbox.pack(side="left", fill="y", padx=10, pady=10)

        self.player_frame = ctk.CTkFrame(self.review_split)
        self.player_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.video_label = ctk.CTkLabel(self.player_frame, text="Select a clip to review")
        self.video_label.pack(fill="both", expand=True)

        self.timeline = ctk.CTkSlider(self.player_frame, from_=0, to=100, command=self.seek_video)
        self.timeline.pack(fill="x", padx=10, pady=5)
        self.timeline.set(0)

        controls = ctk.CTkFrame(self.player_frame)
        controls.pack(fill="x", side="bottom")

        self.play_button = ctk.CTkButton(controls, text="Play", width=80, command=self.toggle_playback)
        self.play_button.pack(side="left", padx=5, pady=5)

        self.speed_selector = ctk.CTkSegmentedButton(
            controls,
            values=["0.5x", "1.0x", "2.0x", "4.0x", "8.0x"],
            command=self.change_speed
        )
        self.speed_selector.set("1.0x")
        self.speed_selector.pack(side="left", padx=10, pady=5)

        ctk.CTkButton(controls, text="Delete False Positive", command=self.delete_current_clip, fg_color="red").pack(side="right", padx=5, pady=5)

        self.current_clip_path = None
        self.clips_buttons = []
        self.current_frame_image = None

    def browse_sample(self):
        path = ctk.filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.mov *.avi *.ts")])
        if path:
            self.sample_path.set(path)
            if not self.input_dir.get():
                self.input_dir.set(os.path.dirname(path))

    def browse_input(self):
        init_dir = self.input_dir.get() or (os.path.dirname(self.sample_path.get()) if self.sample_path.get() else None)
        path = ctk.filedialog.askdirectory(initialdir=init_dir)
        if path:
            self.input_dir.set(path)
            if not self.output_dir.get():
                self.output_dir.set(path)

    def browse_output(self):
        init_dir = self.output_dir.get() or self.input_dir.get() or (os.path.dirname(self.sample_path.get()) if self.sample_path.get() else None)
        path = ctk.filedialog.askdirectory(initialdir=init_dir)
        if path:
            self.output_dir.set(path)

    def log(self, message):
        """Always called on the main thread (directly, or via self.after
        from LogRedirector / background thread)."""
        self.log_box.insert("end", message + "\n")
        self.log_box.see("end")

    def set_progress(self, fraction):
        """Main-thread-safe progress bar update."""
        self.progress_bar.set(fraction)

    def start_processing(self):
        if not self.sample_path.get() or not self.input_dir.get() or not self.output_dir.get():
            self.log("Error: Please fill all directory paths")
            return

        self.is_cancelled = False
        self.stop_button.configure(state="normal")
        self.tabview.set("Progress")
        self.progress_bar.set(0)
        self.update()

        self.log("Opening sample video for ROI selection...")

        cap = cv2.VideoCapture(self.sample_path.get())
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            self.log("Error reading sample frame")
            self.stop_button.configure(state="disabled")
            return

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, "Draw rectangle for ROI. Press SPACE or ENTER to confirm.", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'c' to cancel and abort.", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.namedWindow("Select ROI", cv2.WINDOW_AUTOSIZE)
        cv2.setWindowProperty("Select ROI", cv2.WND_PROP_TOPMOST, 1)
        roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        x, y, w, h = roi
        if w == 0 or h == 0:
            self.log("Empty ROI selected, aborting")
            self.stop_button.configure(state="disabled")
            return
        self.roi_coords = (x, y, x + w, y + h)

        self.log("Starting analysis thread")

        logger = logging.getLogger()
        logger.handlers = []
        logger.setLevel(logging.INFO)
        gui_handler = logging.StreamHandler(LogRedirector(self))
        gui_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(gui_handler)

        file_handler = logging.FileHandler(os.path.join(self.output_dir.get(), "v-dash-scar.log"), encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(file_handler)

        threading.Thread(target=self.run_pipeline, daemon=True).start()

    def run_pipeline(self):
        selected_methods = []
        if self.m1_var.get(): selected_methods.append(1)
        if self.m2_var.get(): selected_methods.append(2)
        if self.m3_var.get(): selected_methods.append(3)

        os.makedirs(self.output_dir.get(), exist_ok=True)
        roi_coords = self.roi_coords

        extensions = [e.strip() for e in self.args.extensions.split(",")]
        video_files = find_video_files(self.input_dir.get(), extensions)

        opts = {
            "frame_step": self.args.frame_step,
            "pixel_diff_thresh": self.args.pixel_diff_thresh,
            "pixel_ratio_thresh": self.args.pixel_ratio_thresh,
            "conf": self.args.conf,
            "merge_sec": self.args.merge_sec,
            "padding_sec": self.args.padding_sec,
            "fast_cut": self.args.fast_cut,
            "verbose_timing": self.args.verbose_timing,
            "bump_detection": self.bump_var.get(),
        }

        resume = bool(self.resume_var.get())
        status = load_status(self.output_dir.get()) if resume else {"methods": {}}

        current_file_list = video_files
        method1_windows = {}

        for method in selected_methods:
            if self.is_cancelled:
                break

            method_out_dir = os.path.join(self.output_dir.get(), f"Method_{method}_Output")
            os.makedirs(method_out_dir, exist_ok=True)

            method_key = str(method)
            prior_results = status.get("methods", {}).get(method_key, {}).get("processed", {})

            to_process = []
            skipped_from_resume = []
            for path in current_file_list:
                if resume and path in prior_results:
                    skipped_from_resume.append(path)
                else:
                    to_process.append(path)

            if skipped_from_resume:
                logging.info(f"Resume: skipping {len(skipped_from_resume)} already-processed files for Method {method}")

            total_files = len(to_process)
            logging.info(f"Starting Method {method} on {total_files} files")

            tasks = []
            for path in to_process:
                windows = method1_windows.get(path) if (method != 1 and method1_windows) else None
                tasks.append((path, method_out_dir, roi_coords, method, windows, opts))

            next_file_list = []
            processed_count = len(skipped_from_resume)
            denom = max(1, total_files + len(skipped_from_resume))

            # Re-seed next_file_list / method1_windows from resumed results
            for path in skipped_from_resume:
                prior = prior_results[path]
                if prior.get("hit"):
                    next_file_list.append(path)
                if method == 1 and prior.get("events"):
                    method1_windows[path] = [tuple(w) for w in prior["events"]]
                if path not in self.unified_db:
                    self.unified_db[path] = {}
                self.unified_db[path][method] = {"hit": prior.get("hit", False)}
                if prior.get("events"):
                    self.unified_db[path]["start"] = prior["events"][0][0]
                    self.unified_db[path]["end"] = prior["events"][-1][1]

            processed_this_method = dict(prior_results) if resume else {}

            if tasks:
                with ProcessPoolExecutor(max_workers=self.args.workers, initializer=init_worker, initargs=(method,)) as executor:
                    futures = {executor.submit(process_video_task, task): task for task in tasks}
                    for future in as_completed(futures):
                        if self.is_cancelled:
                            for f in futures:
                                f.cancel()
                            break

                        processed_count += 1
                        self.after(0, self.set_progress, processed_count / denom)

                        vid_path, hit_found, events, error = future.result()

                        if vid_path not in self.unified_db:
                            self.unified_db[vid_path] = {}
                        self.unified_db[vid_path][method] = {"hit": hit_found}
                        if events:
                            self.unified_db[vid_path]["start"] = events[0][0]
                            self.unified_db[vid_path]["end"] = events[-1][1]
                        if error:
                            self.unified_db[vid_path]["error"] = error

                        processed_this_method[vid_path] = {
                            "hit": hit_found,
                            "events": [list(e) for e in events],
                            "error": error,
                        }

                        if hit_found:
                            next_file_list.append(vid_path)
                        if method == 1:
                            method1_windows[vid_path] = events

            status.setdefault("methods", {})[method_key] = {"processed": processed_this_method}
            save_status(self.output_dir.get(), status)

            current_file_list = sorted(set(next_file_list))
            if not current_file_list or self.is_cancelled:
                break

        if not self.is_cancelled:
            write_unified_report(self.output_dir.get(), self.unified_db)
            logging.info("All tasks completed. Loading incident browser.")
            self.after(0, self.load_review_tab)
        else:
            logging.info("Analysis stopped by user.")

        self.after(0, lambda: self.stop_button.configure(state="disabled"))

    def load_review_tab(self):
        self.tabview.set("Incident Browser")
        for btn in self.clips_buttons:
            btn.destroy()
        self.clips_buttons.clear()

        clip_files = []
        for root_dir, _, files in os.walk(self.output_dir.get()):
            for file in files:
                if file.endswith(".mp4") and "_event_" in file:
                    clip_files.append(os.path.join(root_dir, file))

        for clip in sorted(clip_files):
            name = os.path.basename(clip)
            btn = ctk.CTkButton(self.clip_listbox, text=name, command=lambda c=clip: self.play_clip(c))
            btn.pack(fill="x", pady=2)
            self.clips_buttons.append(btn)

    def change_speed(self, value):
        self.playback_speed = float(value.replace("x", ""))

    def play_clip(self, clip_path):
        if self.play_after_id is not None:
            self.after_cancel(self.play_after_id)
            self.play_after_id = None

        if self.video_cap is not None:
            self.video_cap.release()

        self.current_clip_path = clip_path
        self.video_cap = cv2.VideoCapture(clip_path)

        fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        self.video_fps = fps if (fps and fps > 0) else 30.0

        total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            self.timeline.configure(to=total_frames - 1)
        self.timeline.set(0)

        self.playing = True
        self.play_button.configure(text="Pause")
        self.update_frame()

    def toggle_playback(self):
        if not self.video_cap or not self.video_cap.isOpened():
            return

        self.playing = not self.playing
        if self.playing:
            self.play_button.configure(text="Pause")
            self.update_frame()
        else:
            self.play_button.configure(text="Play")
            if self.play_after_id is not None:
                self.after_cancel(self.play_after_id)
                self.play_after_id = None

    def seek_video(self, value):
        if not self.video_cap or not self.video_cap.isOpened():
            return

        if self.play_after_id is not None:
            self.after_cancel(self.play_after_id)
            self.play_after_id = None

        frame_idx = int(value)
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = self.video_cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 360))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            self.current_frame_image = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 360))
            self.video_label.configure(image=self.current_frame_image, text="")

        if self.playing:
            self.play_after_id = self.after(10, self.update_frame)

    def update_frame(self):
        if self.play_after_id is not None:
            self.after_cancel(self.play_after_id)
            self.play_after_id = None

        if not self.playing or not self.video_cap or not self.video_cap.isOpened():
            return

        start_time = time.perf_counter()

        if self.playback_speed > 1.0:
            skip_count = int(self.playback_speed) - 1
            for _ in range(skip_count):
                self.video_cap.grab()

        ret, frame = self.video_cap.read()

        if ret:
            frame = cv2.resize(frame, (640, 360))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            self.current_frame_image = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 360))
            self.video_label.configure(image=self.current_frame_image, text="")

            current_frame = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.timeline.set(current_frame)
        else:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        render_cost = (time.perf_counter() - start_time) * 1000
        base_interval = 1000.0 / self.video_fps

        if self.playback_speed < 1.0:
            target_interval = base_interval / self.playback_speed
        else:
            target_interval = base_interval

        delay_ms = max(1, int(target_interval - render_cost))

        if self.playing:
            self.play_after_id = self.after(delay_ms, self.update_frame)

    def delete_current_clip(self):
        if self.play_after_id is not None:
            self.after_cancel(self.play_after_id)
            self.play_after_id = None

        self.playing = False
        self.play_button.configure(text="Play")

        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None

        if self.current_clip_path and os.path.exists(self.current_clip_path):
            os.remove(self.current_clip_path)
        self.current_clip_path = None

        # Safe placeholder image, avoids a Tkinter TclError from a stale
        # image reference after the underlying file is gone.
        blank_img = Image.new("RGB", (640, 360), "black")
        self.current_frame_image = ctk.CTkImage(light_image=blank_img, dark_image=blank_img, size=(640, 360))
        self.video_label.configure(image=self.current_frame_image, text="Clip deleted")
        self.timeline.set(0)

        self.load_review_tab()


if __name__ == "__main__":
    args = parse_args()
    if args.version:
        print_banner()
        sys.exit(0)
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    app = VDashScarApp(args)
    app.mainloop()
