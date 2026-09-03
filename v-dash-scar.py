#!/usr/bin/env python3
"""
Vehicle Dashcam Scratch and Collision Automated Recognition (V DASH SCAR).

Author: pr-fuzzylogic
Repo: https://github.com/pr-fuzzylogic/v-dash-scar
Version: 1.1.0
License: MIT (see LICENSE)

A cascade pipeline for triaging large dashcam footage archives to locate
a specific incident such as a scratch, bump, or collision inside a user
defined region of interest, without manually scrubbing through hours of
video. Three detection methods can be run individually or chained so that
each stage filters the file list passed to the next stage. The first
method applies a binary pixel difference test as a fast, high sensitivity
motion filter. The second method applies YOLO bounding box detection
without class filtering. The third method applies YOLO segmentation with
class filtering and precise mask overlap testing, optionally combined
with ByteTrack.

For every method, output is written under a method specific directory
(Method_1_Binary, Method_2_YOLO_BBox, Method_3_YOLO_Seg)
containing the first hit frame as a JPEG with the ROI and detection boxes
drawn on it, an extracted MP4 clip for each detected event, a link or
copy of the source video, a unified report.csv describing hit status and
event timestamps per file, a status.json describing run state for the
resume feature, and a full run log.

Example invocation:
    python v-dash-scar.py --sample sample.mp4 --input ./INPUT_DIR --output ./OUTPUT_DIR --methods 1,2,3

Version 1.1.0 addresses several issues found in the previous iteration.
The mask intersection test previously only checked whether a mask vertex
fell inside the ROI, which produced false negatives whenever the ROI sat
entirely inside a larger mask or a mask edge crossed the ROI without a
vertex landing inside it. It now rasterizes the mask polygon and checks
for a true pixel level overlap against the ROI region. Method 3 reused a
single YOLO model instance with persistent ByteTrack state across every
video handled by a worker process, so tracker history leaked between
unrelated files. The tracker is now reset at the start of every new
video. GUI widgets were previously mutated directly from the background
analysis thread, which is unsafe under Tkinter and customtkinter. All
cross thread UI updates are now marshaled through the main event loop.
The resume flag was previously accepted but never implemented. A
status.json file is now written after each method completes and, when
resume is enabled, is loaded to skip files already processed by a given
method. The methods flag was previously parsed but ignored by the GUI,
which always ran all three methods regardless of the value supplied. The
checkboxes are now pre populated from that flag. Frame seeking through
cv2.CAP_PROP_POS_FRAMES is only approximate on long GOP footage because
the decoder tends to land on the nearest keyframe rather than the exact
requested frame. Window restricted scanning now reads back the actual
landed position after seeking and uses that as the true frame index
baseline, which prevents timestamp drift on long files.

This version restores CoreML acceleration support for Apple Silicon,
selectable inference resolution, a global camera bump filter based on
phase correlation, and frame skipping in the review player so playback
above real time speed no longer stutters. It also removes a race
condition that existed when several worker processes could attempt to
export the same CoreML package at the same time. Export now happens once
in the main process before the worker pool is created, and workers only
ever load an existing package.
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
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed

import customtkinter as ctk
from PIL import Image

import cv2
import numpy as np
import torch
from ultralytics import YOLO

__title__ = "Vehicle Dashcam Scratch and Collision Automated Recognition"
__short_name__ = "V-DASH-SCAR"
__author__ = "pr-fuzzylogic"
__version__ = "1.1.0"
__license__ = "MIT"
__repo__ = "https://github.com/pr-fuzzylogic/v-dash-scar"

DEFAULT_EXTENSIONS = [".mp4", ".mov", ".avi", ".ts"]

# Extra frames appended past the end of a window derived from method one
# when restricting the scanning range of methods two and three. This
# guards against rounding errors when converting seconds to frame indices
# so the real event is not clipped a frame too early.
WINDOW_END_FRAME_BUFFER = 15

# Number of frames before the requested start frame that the decoder is
# asked to seek to first. Reading back the actual landed position after
# that seek lets the frame index bookkeeping stay correct even though long
# GOP footage rarely seeks to the exact requested frame.
SEEK_SAFETY_MARGIN_FRAMES = 5

STATUS_FILENAME = "status.json"

# COCO class identifiers used by the method three class filter. These
# correspond to person, bicycle, car, motorcycle, bus, truck, backpack,
# handbag, suitcase, and skateboard.
METHOD3_CLASSES = [0, 1, 2, 3, 5, 7, 24, 26, 28, 36]

# Magnitude of global phase correlation shift, in pixels on a downscaled
# frame, above which a frame is treated as a camera bump rather than a
# genuine object interaction and is skipped from further analysis.
BUMP_SHIFT_THRESHOLD = 2.0
BUMP_DOWNSCALE_SIZE = (320, 180)

METHOD_DIR_NAMES = {
    1: "Method_1_Binary",
    2: "Method_2_YOLO_BBox",
    3: "Method_3_YOLO_Seg"
}

def print_banner():
    """Prints the tool banner and version information to standard output."""
    banner = f"""
+----------------------------------------------------------------------+
{__short_name__} {__title__}
Version {__version__} Author {__author__} License {__license__}
{__repo__}
+----------------------------------------------------------------------+
Run interactively by launching the script with no arguments.
Run through the command line using the help flag for usage details.
+----------------------------------------------------------------------+
"""
    print(banner, flush=True)


def parse_args():
    """Builds and parses the command line arguments for the tool."""
    parser = argparse.ArgumentParser(
        prog=__short_name__,
        description=f"{__title__} ({__short_name__}) version {__version__}, a dashcam footage triage tool combining motion difference and a YOLO cascade"
    )

    parser.add_argument("--version", action="store_true", help="Print version info and exit")
    parser.add_argument("--sample", type=str, help="Path to sample video file used to pick the region of interest")
    parser.add_argument("--input", type=str, help="Input directory containing video files")
    parser.add_argument("--output", type=str, help="Output directory for results")
    parser.add_argument("--methods", type=str, help="Comma separated methods to run in order, for example 1,2,3")
    parser.add_argument("--extensions", type=str, default=",".join(DEFAULT_EXTENSIONS),
                         help="Comma separated video extensions to scan, case insensitive")
    parser.add_argument("--frame-step", type=int, default=5,
                         help="Analyze every Nth frame, default is 5")
    parser.add_argument("--pixel-diff-thresh", type=int, default=35,
                         help="Method one per pixel grayscale difference threshold, default is 35")
    parser.add_argument("--pixel-ratio-thresh", type=float, default=0.02,
                         help="Method one fraction of ROI pixels that must change to trigger a hit, default is 0.02")
    parser.add_argument("--conf", type=float, default=0.30, help="YOLO confidence threshold, default is 0.30")
    parser.add_argument("--merge-sec", type=float, default=10.0,
                         help="Merge hits into a single event when the gap is below this many seconds, default is 10")
    parser.add_argument("--padding-sec", type=float, default=3.0,
                         help="Extra seconds of padding added before and after each exported clip, default is 3")
    parser.add_argument("--fast-cut", action="store_true",
                         help="Use a fast keyframe based ffmpeg cut without re encoding, less precise")
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel worker processes, default is 2")
    parser.add_argument("--bump-detection", action="store_true", help="Enable a global camera bump filter based on phase correlation")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution used by the YOLO models")
    parser.add_argument("--coreml", action="store_true", help="Use CoreML hardware acceleration on macOS")
    parser.add_argument("--resume", action="store_true",
                         help="Skip files already recorded in a previous run's status file for the same method")
    parser.add_argument("--verbose-timing", action="store_true",
                         help="Log per file decode and inference timing diagnostics in addition to the per frame average summary")
    return parser.parse_args()


def check_bbox_intersection(box, roi):
    """Tests whether an axis aligned detection box overlaps the ROI rectangle."""
    bx1, by1, bx2, by2 = box
    rx1, ry1, rx2, ry2 = roi
    return not (bx2 < rx1 or bx1 > rx2 or by2 < ry1 or by1 > ry2)


def check_mask_intersection(mask_points, roi, frame_shape):
    """Performs a true polygon to rectangle intersection test.

    Testing only the polygon vertices against the ROI rectangle produces
    false negatives whenever the ROI sits entirely inside a larger mask,
    since no vertex ever lands inside it, or whenever a mask edge crosses
    the ROI without any vertex landing inside it. This function rasterizes
    the mask polygon onto a canvas the size of the frame and checks for a
    real pixel level overlap against the ROI region, which is a correct
    intersection test at a modest extra cost.
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
    """Expands each start and end window by the padding value and merges overlaps."""
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


def find_video_files(input_dir, extensions, output_dir=None):
    """Recursively discovers video files matching the requested extensions.

    Extension comparison is case insensitive so names such as an uppercase
    variant of a common extension are not silently missed. Automatically skips
    configured output directories to prevent analyzing previously generated clips.
    """
    wanted = {e.lower() for e in extensions}
    matches = []

    skip_dirs = []
    if output_dir:
        for m, dname in METHOD_DIR_NAMES.items():
            skip_dirs.append(os.path.abspath(os.path.join(output_dir, dname)))

    for root, dirs, files in os.walk(input_dir):
        dirs[:] = [d for d in dirs if os.path.abspath(os.path.join(root, d)) not in skip_dirs]
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in wanted:
                matches.append(os.path.join(root, name))
    return sorted(set(matches))


def load_status(output_dir):
    """Loads the resume status file, returning an empty structure if absent or invalid."""
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
    """Writes the resume status file atomically using a temporary file and rename."""
    path = os.path.join(output_dir, STATUS_FILENAME)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
    os.replace(tmp_path, path)


_WORKER_MODEL = None
_WORKER_DEVICE = None


def get_device():
    """Selects the best available compute backend, preferring CUDA, then MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def coreml_package_path(model_prefix, imgsz):
    """Builds the expected CoreML package path for a given model and resolution."""
    return f"{model_prefix}_{imgsz}.mlpackage"


def ensure_coreml_export(pt_path, model_prefix, imgsz, task):
    """Exports a CoreML package once, from the main process, before any workers start.

    Earlier revisions attempted this export from inside each worker's
    initializer. Since a process pool starts several workers at once, they
    could all attempt to export the same package concurrently, which is a
    race condition that risks a corrupted or partially written package.
    Performing the export here, in the main process, sequentially and
    before the pool is created, removes that race entirely. Workers only
    ever load an already exported package.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_pt_path = os.path.join(script_dir, pt_path)
    target_path = os.path.join(script_dir, coreml_package_path(model_prefix, imgsz))

    if os.path.exists(target_path):
        return target_path

    lock_path = target_path + ".lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        logging.warning(f"Export lock already present at {lock_path}, waiting for an existing export to finish")
        while os.path.exists(lock_path) and not os.path.exists(target_path):
            time.sleep(1.0)
        if os.path.exists(target_path):
            return target_path

    try:
        logging.info(f"Exporting {abs_pt_path} to CoreML at resolution {imgsz}, this runs once and is then reused")
        model = YOLO(abs_pt_path)
        exported_path = model.export(format="coreml", imgsz=imgsz, nms=True)

        if exported_path and os.path.exists(exported_path) and exported_path != target_path:
            if os.path.exists(target_path):
                shutil.rmtree(target_path)
            shutil.move(exported_path, target_path)
    finally:
        if os.path.exists(lock_path):
            os.remove(lock_path)

    return target_path


def init_worker(method, use_coreml=False, imgsz=640):
    """Runs once per worker process when the pool for a given method is created.

    Workers only load models here. CoreML export, when requested, has
    already completed in the main process before the pool was created, so
    there is no export race between workers.
    """
    global _WORKER_MODEL, _WORKER_DEVICE
    _WORKER_DEVICE = get_device()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if method == 2:
        base_model = os.path.join(script_dir, "yolo11n.pt")
        model_prefix = "yolo11n"
        task = "detect"
    elif method == 3:
        base_model = os.path.join(script_dir, "yolo11n-seg.pt")
        model_prefix = "yolo11n_seg"
        task = "segment"
    else:
        _WORKER_MODEL = None
        logging.info(f"Worker ready for method {method} on device {_WORKER_DEVICE}, process id {os.getpid()}")
        return

    if use_coreml and platform.system() == "Darwin":
        coreml_path = os.path.join(script_dir, coreml_package_path(model_prefix, imgsz))
        _WORKER_MODEL = YOLO(coreml_path, task=task)
        _WORKER_DEVICE = "cpu"
    else:
        _WORKER_MODEL = YOLO(base_model)

    logging.info(f"Worker ready for method {method} on device {_WORKER_DEVICE}, process id {os.getpid()}")


def reset_tracker_state(model):
    """Forces the tracker to be reinitialized on the next call.

    A model instance reused across multiple video files within the same
    worker process, with persistent tracking enabled, would otherwise
    carry track history from the previous file into the next one.
    """
    if model is not None:
        model.predictor = None


def extract_clip(video_path, start_sec, end_sec, padding, output_path, fast_cut=False):
    """Extracts a padded clip around an event using ffmpeg.

    The seek flag is placed before the input flag in both branches. This
    makes ffmpeg seek directly to the nearest keyframe at the container
    level instead of decoding the stream from the beginning, which matters
    for long dashcam files since otherwise every cut on an hour long file
    would decode the whole hour first. The fast cut path copies the stream
    without re encoding, which is quicker but not frame accurate. The
    default path re encodes only the short requested window, which is
    slower but frame accurate and preferable for evidence purposes.
    """
    clip_start = max(0.0, start_sec - padding)
    clip_end = end_sec + padding
    duration = clip_end - clip_start

    if fast_cut:
        cmd = ["ffmpeg", "-y", "-nostdin",
               "-ss", str(clip_start), "-i", video_path,
               "-t", str(duration), "-c", "copy", output_path]
    else:
        cmd = ["ffmpeg", "-y", "-nostdin",
               "-ss", str(clip_start), "-i", video_path, "-t", str(duration),
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
               "-c:a", "aac", output_path]

    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        logging.warning(f"ffmpeg failed for {output_path}: {result.stderr.decode(errors='ignore')[:300]}")
        return False
    return True


def iter_frames(cap, fps, frame_step, windows=None):
    """Yields frame index and frame pairs, optionally restricted to given time windows.

    Restricting to windows derived from method one drastically reduces the
    time spent in methods two and three once candidate segments are known,
    since the capture seeks directly to each window instead of decoding
    the whole file. Seeking through the frame position property is only
    approximate on long GOP footage, since the decoder typically lands on
    the nearest keyframe rather than the exact requested frame. To avoid
    silent timestamp drift, the capture is seeked to a position slightly
    before the requested start, and the actual landed position is read
    back and used as the true frame index baseline instead of trusting the
    request blindly.
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


def link_or_copy(src, dst):
    """Links the source file into the destination, falling back to a copy when needed."""
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
        logging.warning(f"Could not link or copy {src} to {dst}: {e}")


def process_video_task(task):
    """Processes a single video file for one method, returning hit status and events."""
    (video_path, output_dir, roi_coords, method, restrict_windows, opts) = task

    base_name = os.path.basename(video_path)
    rx1, ry1, rx2, ry2 = roi_coords
    roi_area = max(1, (rx2 - rx1) * (ry2 - ry1))

    cap = None
    try:
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

            # Global camera bump detection compares the whole downscaled
            # frame against the previous one using phase correlation. A
            # large global shift indicates the camera itself moved, for
            # example from a physical impact on the vehicle, rather than
            # an object moving within an otherwise stable frame, so the
            # frame is skipped from the object level analysis below.
            bump_detected = False
            if bump_detection:
                gray_frame_cache = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                small_gray = cv2.resize(gray_frame_cache, BUMP_DOWNSCALE_SIZE)
                curr_global_gray = np.float32(small_gray)

                if prev_global_gray is not None:
                    shift, response = cv2.phaseCorrelate(prev_global_gray, curr_global_gray)
                    magnitude = np.sqrt(shift[0] ** 2 + shift[1] ** 2)
                    if magnitude > BUMP_SHIFT_THRESHOLD:
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
                # This method is intentionally not class filtered. A
                # wheelie bin, a person, or a cleaning cart can all be the
                # relevant object, and filtering by class risks missing
                # the real event, for example an object rolling into frame
                # before a person appears.
                results = model.predict(source=frame, conf=opts["conf"], imgsz=opts["imgsz"],
                                         device=device, verbose=False)
                if results[0].boxes is not None:
                    xyxy = results[0].boxes.xyxy
                    boxes = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else np.array(xyxy)
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
                results = model.track(source=frame, conf=opts["conf"], imgsz=opts["imgsz"],
                                       classes=METHOD3_CLASSES,
                                       persist=True, tracker="bytetrack.yaml",
                                       device=device, verbose=False)
                if results[0].boxes is not None and results[0].masks is not None:
                    xyxy = results[0].boxes.xyxy
                    boxes = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else np.array(xyxy)
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
                    f"DIAGNOSTICS {base_name} frames {frames_processed} "
                    f"decode total {total_decode_time:.2f}s inference total {total_inference_time:.2f}s "
                    f"average decode {avg_decode_ms:.1f}ms per frame average inference {avg_inference_ms:.1f}ms per frame"
                )
            else:
                logging.info(
                    f"TIMING {base_name} frames {frames_processed} "
                    f"average decode {avg_decode_ms:.1f}ms per frame average inference {avg_inference_ms:.1f}ms per frame"
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
    finally:
        if cap is not None:
            cap.release()


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

def resize_with_aspect_ratio(frame, target_width=640, target_height=360):
    h, w = frame.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    top = (target_height - new_h) // 2
    bottom = target_height - new_h - top
    left = (target_width - new_w) // 2
    right = target_width - new_w - left
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))


class LogRedirector:
    """Redirects standard logging output into the GUI log widget on the main thread."""

    def __init__(self, app):
        self.app = app

    def write(self, message):
        if message.strip():
            # This may be called from the background analysis thread, and
            # Tkinter based widgets must only be touched from the main
            # thread, so the update is scheduled through the event loop.
            self.app.after(0, self.app.log, message)

    def flush(self):
        pass


class VDashScarApp(ctk.CTk):
    """Main application window covering configuration, progress, and review."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.title(f"{__short_name__} version {__version__}")
        self.geometry("1000x700")

        self.sample_path = ctk.StringVar(value=args.sample if args.sample else "")
        self.input_dir = ctk.StringVar(value=args.input if args.input else "")
        self.output_dir = ctk.StringVar(value=args.output if args.output else "")

        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)

        self.tab_config = self.tabview.add("Configuration")
        self.tab_progress = self.tabview.add("Progress")
        self.tab_review = self.tabview.add("Incident Browser")

        # The methods flag was previously parsed but ignored by the GUI,
        # which always ran all three methods regardless of its value. The
        # checkboxes are now pre populated from this flag when supplied.
        requested_methods = {1, 2, 3}
        if args.methods:
            try:
                requested_methods = {int(m.strip()) for m in args.methods.split(",") if m.strip()}
            except ValueError:
                logging.warning(f"Could not parse methods value {args.methods!r}, defaulting to all three")
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
        self.entry_sample = ctk.CTkEntry(frame, textvariable=self.sample_path, width=400)
        self.entry_sample.grid(row=0, column=1, padx=10, pady=10)
        self.btn_browse_sample = ctk.CTkButton(frame, text="Browse", command=self.browse_sample)
        self.btn_browse_sample.grid(row=0, column=2, padx=10, pady=10)

        ctk.CTkLabel(frame, text="Input Directory").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.entry_input = ctk.CTkEntry(frame, textvariable=self.input_dir, width=400)
        self.entry_input.grid(row=1, column=1, padx=10, pady=10)
        self.btn_browse_input = ctk.CTkButton(frame, text="Browse", command=self.browse_input)
        self.btn_browse_input.grid(row=1, column=2, padx=10, pady=10)

        ctk.CTkLabel(frame, text="Output Directory").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.entry_output = ctk.CTkEntry(frame, textvariable=self.output_dir, width=400)
        self.entry_output.grid(row=2, column=1, padx=10, pady=10)
        self.btn_browse_output = ctk.CTkButton(frame, text="Browse", command=self.browse_output)
        self.btn_browse_output.grid(row=2, column=2, padx=10, pady=10)

        self.m1_var = ctk.BooleanVar(value=1 in requested_methods)
        self.m2_var = ctk.BooleanVar(value=2 in requested_methods)
        self.m3_var = ctk.BooleanVar(value=3 in requested_methods)

        self.chk_m1 = ctk.CTkCheckBox(frame, text="Method 1 Binary pixel difference", variable=self.m1_var)
        self.chk_m1.grid(row=3, column=0, columnspan=3, padx=10, pady=5, sticky="w")
        self.chk_m2 = ctk.CTkCheckBox(frame, text="Method 2 YOLO bounding boxes", variable=self.m2_var)
        self.chk_m2.grid(row=4, column=0, columnspan=3, padx=10, pady=5, sticky="w")
        self.chk_m3 = ctk.CTkCheckBox(frame, text="Method 3 YOLO segmentation", variable=self.m3_var)
        self.chk_m3.grid(row=5, column=0, columnspan=3, padx=10, pady=5, sticky="w")

        self.resume_var = ctk.BooleanVar(value=bool(self.args.resume))
        self.chk_resume = ctk.CTkCheckBox(frame, text="Resume by skipping files already recorded in status.json", variable=self.resume_var)
        self.chk_resume.grid(row=6, column=0, columnspan=3, padx=10, pady=5, sticky="w")

        self.bump_var = ctk.BooleanVar(value=bool(self.args.bump_detection))
        self.chk_bump = ctk.CTkCheckBox(frame, text="Enable camera bump detection", variable=self.bump_var)
        self.chk_bump.grid(row=7, column=0, columnspan=3, padx=10, pady=5, sticky="w")

        self.imgsz_var = ctk.StringVar(value=str(getattr(self.args, "imgsz", 640)))
        ctk.CTkLabel(frame, text="Inference Resolution").grid(row=8, column=0, padx=10, pady=5, sticky="w")
        self.seg_imgsz = ctk.CTkSegmentedButton(frame, variable=self.imgsz_var, values=["640", "1280"])
        self.seg_imgsz.grid(row=8, column=1, padx=10, pady=5, sticky="w")

        self.coreml_var = ctk.BooleanVar(value=bool(getattr(self.args, "coreml", False)))
        self.chk_coreml = None
        if platform.system() == "Darwin":
            self.chk_coreml = ctk.CTkCheckBox(frame, text="Use CoreML acceleration on the Apple Neural Engine", variable=self.coreml_var)
            self.chk_coreml.grid(row=9, column=0, columnspan=3, padx=10, pady=5, sticky="w")

        self.start_button = ctk.CTkButton(frame, text="Start Analysis", command=self.start_processing, fg_color="green")
        self.start_button.grid(row=10, column=0, columnspan=3, pady=(30, 5))

        self.start_status_label = ctk.CTkLabel(frame, text="", text_color="gray", font=("", 11))
        self.start_status_label.grid(row=11, column=0, columnspan=3, pady=(0, 10))

    def build_progress_tab(self):
        self.log_box = ctk.CTkTextbox(self.tab_progress, state="normal", wrap="word")
        self.log_box.pack(fill="both", expand=True, padx=10, pady=10)

        self.progress_bar = ctk.CTkProgressBar(self.tab_progress)
        self.progress_bar.pack(fill="x", padx=10, pady=10)
        self.progress_bar.set(0)

        self.stop_button = ctk.CTkButton(self.tab_progress, text="Stop Analysis", command=self.stop_processing, fg_color="red", state="disabled")
        self.stop_button.pack(pady=5)

        self.is_cancelled = False
        self.is_pipeline_running = False

    def stop_processing(self):
        self.is_cancelled = True
        self.log("Stopping analysis, waiting for active tasks to finish")
        self.stop_button.configure(state="disabled")
        self.wait_for_cancel_loop()

    def set_config_ui_state(self, state_str):
        """Locks or unlocks the configuration tab widgets to prevent mid-run changes."""
        self.entry_sample.configure(state=state_str)
        self.btn_browse_sample.configure(state=state_str)
        self.entry_input.configure(state=state_str)
        self.btn_browse_input.configure(state=state_str)
        self.entry_output.configure(state=state_str)
        self.btn_browse_output.configure(state=state_str)
        self.chk_m1.configure(state=state_str)
        self.chk_m2.configure(state=state_str)
        self.chk_m3.configure(state=state_str)
        self.chk_resume.configure(state=state_str)
        self.chk_bump.configure(state=state_str)
        self.seg_imgsz.configure(state=state_str)
        if self.chk_coreml is not None:
            self.chk_coreml.configure(state=state_str)
        self.start_button.configure(state=state_str)

    def wait_for_cancel_loop(self):
        if self.is_pipeline_running:
            self.log_box.insert("end", ".")
            self.log_box.see("end")
            self.start_status_label.configure(text="Stopping active tasks, the Start button will unlock automatically...")
            self.after(1000, self.wait_for_cancel_loop)

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
                self.check_resume_availability(path)

    def browse_output(self):
        init_dir = self.output_dir.get() or self.input_dir.get() or (os.path.dirname(self.sample_path.get()) if self.sample_path.get() else None)
        path = ctk.filedialog.askdirectory(initialdir=init_dir)
        if path:
            self.output_dir.set(path)
            self.check_resume_availability(path)

    def check_resume_availability(self, path):
        if os.path.exists(os.path.join(path, STATUS_FILENAME)):
            self.resume_var.set(True)
            self.log("Status file found in the output directory. Resume option automatically enabled.")

    def log(self, message):
        """Appends a message to the log widget. Always runs on the main thread."""
        self.log_box.insert("end", message + "\n")
        self.log_box.see("end")

    def set_progress(self, fraction):
        """Updates the progress bar. Always runs on the main thread."""
        self.progress_bar.set(fraction)

    def start_processing(self):
        if getattr(self, "is_pipeline_running", False):
            self.log("Previous analysis is still stopping, please wait.")
            return

        if not self.sample_path.get() or not self.input_dir.get() or not self.output_dir.get():
            self.log("Error, please fill all directory paths")
            return

        self.set_config_ui_state("disabled")
        self.start_status_label.configure(text="Analysis starting - select ROI in the OpenCV window")
        self.update()

        self.is_cancelled = False
        self.is_pipeline_running = True
        self.stop_button.configure(state="normal")
        self.tabview.set("Progress")
        self.progress_bar.set(0)
        self.update()

        self.log("Opening sample video for ROI selection.")

        cap = cv2.VideoCapture(self.sample_path.get())
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            self.log("Error reading sample frame")
            self.stop_button.configure(state="disabled")
            self.set_config_ui_state("normal")
            self.start_status_label.configure(text="")
            self.is_pipeline_running = False
            return

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, "Draw a rectangle for the ROI, press space or enter to confirm.", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Press c to cancel and abort.", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.namedWindow("Select ROI", cv2.WINDOW_AUTOSIZE)
        cv2.setWindowProperty("Select ROI", cv2.WND_PROP_TOPMOST, 1)
        roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        x, y, w, h = roi
        if w == 0 or h == 0:
            self.log("Empty ROI selected, aborting")
            self.stop_button.configure(state="disabled")
            self.set_config_ui_state("normal")
            self.start_status_label.configure(text="")
            self.is_pipeline_running = False
            return

        self.start_status_label.configure(text="Analysis in progress...")
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
        if self.m1_var.get():
            selected_methods.append(1)
        if self.m2_var.get():
            selected_methods.append(2)
        if self.m3_var.get():
            selected_methods.append(3)

        os.makedirs(self.output_dir.get(), exist_ok=True)
        roi_coords = self.roi_coords

        extensions = [e.strip() for e in self.args.extensions.split(",")]
        video_files = find_video_files(self.input_dir.get(), extensions, self.output_dir.get())

        imgsz = int(self.imgsz_var.get())
        use_coreml = bool(self.coreml_var.get())

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
            "imgsz": imgsz,
            "coreml": use_coreml,
        }

        resume = bool(self.resume_var.get())
        status = load_status(self.output_dir.get()) if resume else {"methods": {}}

        current_file_list = video_files
        method1_windows = {}

        for method in selected_methods:
            if self.is_cancelled:
                break

            # CoreML export, when requested, happens once here in the main
            # process before the worker pool is created for this method.
            # This removes the race condition that existed when several
            # workers could attempt to export the same package at once.
            if use_coreml and platform.system() == "Darwin":
                if method == 2:
                    ensure_coreml_export("yolo11n.pt", "yolo11n", imgsz, "detect")
                elif method == 3:
                    ensure_coreml_export("yolo11n-seg.pt", "yolo11n_seg", imgsz, "segment")

            method_out_dir = os.path.join(self.output_dir.get(), METHOD_DIR_NAMES[method])
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
                logging.info(f"Resume, skipping {len(skipped_from_resume)} already processed files for method {method}")

            total_files = len(to_process)
            logging.info(f"Starting method {method} on {total_files} files")

            tasks = []
            for path in to_process:
                windows = method1_windows.get(path) if (method != 1 and method1_windows) else None
                tasks.append((path, method_out_dir, roi_coords, method, windows, opts))

            next_file_list = []
            processed_count = len(skipped_from_resume)
            denom = max(1, total_files + len(skipped_from_resume))

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
                with ProcessPoolExecutor(max_workers=self.args.workers, initializer=init_worker, initargs=(method, use_coreml, imgsz)) as executor:
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

                        file_base = os.path.basename(vid_path)
                        logging.info(f"Method {method} processed file {processed_count} of {denom} {file_base}")

            status.setdefault("methods", {})[method_key] = {"processed": processed_this_method}
            save_status(self.output_dir.get(), status)

            current_file_list = sorted(set(next_file_list))
            if not current_file_list or self.is_cancelled:
                break

        write_unified_report(self.output_dir.get(), self.unified_db)

        if not self.is_cancelled:
            logging.info("All tasks completed, loading incident browser.")
        else:
            logging.info("\nAnalysis stopped by user, active tasks finished.")
            logging.info("Partial results saved and available in the incident browser.")
            logging.info("Enable the resume option in configuration to continue from this point later.")

        self.is_pipeline_running = False
        self.after(0, self.load_review_tab)
        self.after(0, lambda: self.stop_button.configure(state="disabled"))
        self.after(0, lambda: self.set_config_ui_state("normal"))
        self.after(0, lambda: self.start_status_label.configure(text=""))

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

        self.clip_paths = sorted(clip_files)
        for clip in self.clip_paths:
            name = os.path.basename(clip)
            btn = ctk.CTkButton(self.clip_listbox, text=name, command=lambda c=clip: self.play_clip(c))
            btn.pack(fill="x", pady=2)
            self.clips_buttons.append(btn)

    def play_next_clip(self):
        if not self.current_clip_path or not hasattr(self, "clip_paths") or not self.clip_paths:
            return
        try:
            idx = self.clip_paths.index(self.current_clip_path)
            next_idx = (idx + 1) % len(self.clip_paths)
            self.play_clip(self.clip_paths[next_idx])
        except ValueError:
            pass

    def change_speed(self, value):
        self.playback_speed = float(value.replace("x", ""))

    def update_frame(self):
        if self.play_after_id is not None:
            self.after_cancel(self.play_after_id)
            self.play_after_id = None

        if not self.playing or not self.video_cap or not self.video_cap.isOpened():
            return

        start_time = time.perf_counter()

        if not hasattr(self, 'last_frame_time'):
            self.last_frame_time = start_time

        # Target interval calculation per frame at current speed
        target_interval_sec = (1.0 / self.video_fps) / self.playback_speed
        elapsed_sec = start_time - self.last_frame_time

        # Calculate frames to skip to maintain real time synchronization
        frames_to_skip = int(elapsed_sec / target_interval_sec)

        if self.playback_speed > 1.0:
            # Enforce minimum frame skip threshold for fast playback
            frames_to_skip = max(frames_to_skip, int(self.playback_speed) - 1)

        for _ in range(frames_to_skip):
            self.video_cap.grab()

        ret, frame = self.video_cap.read()

        if ret:
            frame = resize_with_aspect_ratio(frame, 640, 360)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            self.current_frame_image = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 360))
            self.video_label.configure(image=self.current_frame_image, text="")

            current_frame = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.timeline.set(current_frame)

            # Time tracker adjustment incorporating skipped frames to prevent cumulative timing error
            self.last_frame_time += (frames_to_skip + 1) * target_interval_sec

            # Failsafe reset for extreme system stalls
            if start_time - self.last_frame_time > 0.5:
                 self.last_frame_time = start_time
        else:
            self.play_next_clip()
            return

        render_cost = (time.perf_counter() - start_time)

        # Next call delay derived from target interval minus render cost
        delay_sec = target_interval_sec - render_cost
        delay_ms = max(1, int(delay_sec * 1000))

        if self.playing:
            self.play_after_id = self.after(delay_ms, self.update_frame)

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

        # Time tracking reset on clip change
        if hasattr(self, 'last_frame_time'):
            delattr(self, 'last_frame_time')

        self.playing = True
        self.play_button.configure(text="Pause")
        self.update_frame()

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
            frame = resize_with_aspect_ratio(frame, 640, 360)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            self.current_frame_image = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 360))
            self.video_label.configure(image=self.current_frame_image, text="")

        # Time tracking reset on seek
        if hasattr(self, 'last_frame_time'):
            delattr(self, 'last_frame_time')

        if self.playing:
            self.play_after_id = self.after(10, self.update_frame)

    def toggle_playback(self):
        if not self.video_cap or not self.video_cap.isOpened():
            return

        self.playing = not self.playing
        if self.playing:
            self.play_button.configure(text="Pause")
            # Time tracking reset on unpause
            if hasattr(self, 'last_frame_time'):
                delattr(self, 'last_frame_time')
            self.update_frame()
        else:
            self.play_button.configure(text="Play")
            if self.play_after_id is not None:
                self.after_cancel(self.play_after_id)
                self.play_after_id = None

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

        # A placeholder image avoids a Tkinter error caused by a stale
        # image reference after the underlying file has been removed.
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
