#!/usr/bin/env python3
"""
============================================================================
 Vehicle Dashcam Scratch & Collision Automated Recognition (V-DASH-SCAR)
============================================================================

 Author:   pr-fuzzylogic
 Repo:     https://github.com/pr-fuzzylogic/vehicle-dashcam-scratch-collision-automated-recognition
 Version:  1.0.0
 License:  MIT (see LICENSE)

 Description:
   Cascade pipeline for triaging large dashcam footage archives to locate
   a specific incident (e.g. a scratch, bump, or collision) inside a
   user-defined region of interest (ROI), without manually scrubbing
   through hours/terabytes of video.

   Methods (run individually or chained; each stage filters the file list
   passed to the next):
     1. Binary pixel difference   - fast, high-sensitivity motion filter
     2. YOLO bounding boxes       - object detection, no class filtering
     3. YOLO segmentation         - class-filtered, precise mask overlap

 Output (per method, under <output_dir>/Method_X_.../):
     *_hit.jpg         first frame where a hit was detected (ROI + boxes drawn)
     *_event_N.mp4     extracted clip for each detected event
     <original file>   symlink/hardlink/copy to the source video
     report.csv        per-file status (hit/clean/error) and event timestamps
     status.json       run state, used for --resume
     v-dash-scar.log   full run log

 Usage example:
     python v-dash-scar.py --sample sample.mp4 --input ./INPUT_DIR --output ./OUTPUT_DIR --methods 1,2,3
============================================================================
"""

__title__ = "Vehicle Dashcam Scratch & Collision Automated Recognition"
__short_name__ = "V-DASH-SCAR"
__author__ = "pr-fuzzylogic"
__version__ = "1.0.0"
__license__ = "MIT"
__repo__ = "https://github.com/pr-fuzzylogic/vehicle-dashcam-scratch-collision-automated-recognition"

import os
import sys
import json
import csv
import shutil
import logging
import argparse
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import torch
import time
from ultralytics import YOLO

DEFAULT_EXTENSIONS = [".mp4", ".mov", ".avi", ".ts"]

# Extra frames appended to the end of a Method-1-derived window when
# restricting Methods 2/3 scanning range, to protect against rounding
# errors when converting seconds -> frame indices (avoids clipping the
# real event by one frame too early).
WINDOW_END_FRAME_BUFFER = 15


# ============================================================================
# Banner and version info
# ============================================================================

def print_banner():
    banner = f"""
+----------------------------------------------------------------------+
  {__short_name__} - {__title__}
  Version {__version__}  |  Author: {__author__}  |  License: {__license__}
  {__repo__}
+----------------------------------------------------------------------+
  Run interactively: ./v-dash-scar.py
  Run via CLI:        ./v-dash-scar.py --help
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


def check_mask_intersection(mask_points, roi):
    rx1, ry1, rx2, ry2 = roi
    for px, py in mask_points:
        if rx1 <= px <= rx2 and ry1 <= py <= ry2:
            return True
    return False


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

    A small frame buffer (WINDOW_END_FRAME_BUFFER) is added past the
    computed end frame to guard against float rounding when converting
    seconds -> frame index, so real events aren't clipped a frame early."""
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
            start_frame = max(0, int(w_start * fps))
            end_frame = int(w_end * fps) + 1 + WINDOW_END_FRAME_BUFFER
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_idx = start_frame
            while frame_idx <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_step == 0:
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
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return video_path, False, [], "Could not open file"

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        frame_step = opts["frame_step"]
        merge_threshold_sec = opts["merge_sec"]
        padding_sec = opts["padding_sec"]
        verbose_timing = opts.get("verbose_timing", False)

        windows = merge_windows(restrict_windows, padding_sec) if restrict_windows else None

        events = []
        current_event_start = None
        current_event_end = None
        first_hit_frame_saved = False
        hit_found_in_file = False
        prev_roi_gray = None

        model = _WORKER_MODEL
        device = _WORKER_DEVICE

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

            inference_start = time.perf_counter()

            if method == 1:
                curr_roi_crop = frame[ry1:ry2, rx1:rx2]
                if curr_roi_crop.size > 0:
                    curr_roi_gray = cv2.cvtColor(curr_roi_crop, cv2.COLOR_BGR2GRAY)
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
                # Intentionally NOT class-filtered: a wheelie bin, a person, a
                # cleaning cart etc. can all be the relevant object, and
                # filtering by class risks missing the real event (e.g. an
                # object rolling into frame before the person appears).
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
                                       classes=[0, 1, 2, 3, 5, 7, 24, 26, 28, 36],
                                       persist=True, tracker="bytetrack.yaml",
                                       device=device, verbose=False)
                if results[0].boxes is not None and results[0].masks is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    masks = results[0].masks.xy
                    for box, mask in zip(boxes, masks):
                        if check_bbox_intersection(box, roi_coords) and check_mask_intersection(mask, roi_coords):
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


def write_report(method_out_dir, rows):
    csv_path = os.path.join(method_out_dir, "report.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "hit", "event_start_sec", "event_end_sec", "error"])
        for row in rows:
            writer.writerow(row)


def load_status(method_out_dir):
    status_path = os.path.join(method_out_dir, "status.json")
    if os.path.exists(status_path):
        with open(status_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_status(method_out_dir, status):
    status_path = os.path.join(method_out_dir, "status.json")
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)


def gui_select_methods(root):
    root.deiconify()
    root.title(f"{__short_name__} - Select Analysis Methods")

    tk.Label(root, text="Select methods to run sequentially").pack(anchor="w", padx=10, pady=5)

    var1 = tk.BooleanVar(value=True)
    var2 = tk.BooleanVar(value=True)
    var3 = tk.BooleanVar(value=True)

    tk.Checkbutton(root, text="Method 1: Binary pixel difference (fast, sensitive)", variable=var1).pack(anchor="w", padx=10)
    tk.Checkbutton(root, text="Method 2: YOLO bounding boxes, no class filter", variable=var2).pack(anchor="w", padx=10)
    tk.Checkbutton(root, text="Method 3: YOLO segmentation, class-filtered, precise shape", variable=var3).pack(anchor="w", padx=10)

    selected_methods = []

    def on_confirm():
        if var1.get():
            selected_methods.append(1)
        if var2.get():
            selected_methods.append(2)
        if var3.get():
            selected_methods.append(3)
        root.destroy()

    tk.Button(root, text="Confirm", command=on_confirm).pack(pady=10)
    root.mainloop()
    return selected_methods


# ============================================================================
# Main pipeline
# ============================================================================

def main():
    print_banner()
    args = parse_args()

    if args.version:
        print(f"{__title__} ({__short_name__}) v{__version__} by {__author__}")
        print(__repo__)
        return

    sample_video_path = args.sample
    input_dir = args.input
    output_dir = args.output
    selected_methods = [int(m.strip()) for m in args.methods.split(",")] if args.methods else []
    extensions = [e.strip() for e in args.extensions.split(",")]

    if not sample_video_path or not input_dir or not output_dir or not selected_methods:
        root = tk.Tk()
        root.withdraw()

        # Force macOS to render dialogs in front of the terminal
        root.attributes('-topmost', True)

        print("Interactive mode started. Check the popup windows.", flush=True)

        messagebox.showinfo(
            "Step 1 of 4: Sample Video",
            "Please select a SAMPLE VIDEO FILE.\n\n"
            "This video will be used to draw the Region of Interest (ROI) where the tool should look for events. "
            "Select a ROI and then press SPACE or ENTER. Cancel the selection by pressing 'c'.",
            parent=root
        )

        sample_video_path = filedialog.askopenfilename(
            title="Select sample video file",
            filetypes=[("Video files", "*.mp4 *.MP4 *.mov *.MOV *.avi *.AVI")]
        )
        if not sample_video_path:
            return

        base_dir = os.path.dirname(sample_video_path)

        messagebox.showinfo(
            "Step 2 of 4: Input Directory",
            "Please select the INPUT DIRECTORY.\n\n"
            "This folder contains all the dashcam videos you want to analyze.",
            parent=root
        )

        input_dir = filedialog.askdirectory(
            title="Select input directory with video files",
            initialdir=base_dir
        )
        if not input_dir:
            return

        messagebox.showinfo(
            "Step 3 of 4: Output Directory",
            "Please select the OUTPUT DIRECTORY.\n\n"
            "This is where the result clips, frames, and logs will be saved.",
            parent=root
        )

        output_dir = filedialog.askdirectory(
            title="Select output directory",
            initialdir=input_dir
        )
        if not output_dir:
            return

        # Step 4 is handled by the custom GUI which already has labels
        selected_methods = gui_select_methods(root)
        if not selected_methods:
            return

    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "v-dash-scar.log"), encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info(f"{__short_name__} v{__version__} starting run")
    logging.info(f"Device available: {get_device()}")

    cap = cv2.VideoCapture(sample_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        logging.error("Error reading sample frame")
        return

    print("Draw rectangle then press SPACE or ENTER to confirm", flush=True)

    # Force OpenCV window to foreground
    cv2.namedWindow("Select ROI", cv2.WINDOW_AUTOSIZE)
    cv2.setWindowProperty("Select ROI", cv2.WND_PROP_TOPMOST, 1)

    roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    x, y, w, h = roi
    if w == 0 or h == 0:
        logging.error("Empty ROI selected, aborting")
        return
    roi_coords = (x, y, x + w, y + h)

    video_files = find_video_files(input_dir, extensions)
    logging.info(f"Found {len(video_files)} video files under {input_dir}")

    method_names = {
        1: "Method_1_Binary",
        2: "Method_2_YOLO_BBox",
        3: "Method_3_YOLO_Seg",
    }

    opts = {
        "frame_step": args.frame_step,
        "pixel_diff_thresh": args.pixel_diff_thresh,
        "pixel_ratio_thresh": args.pixel_ratio_thresh,
        "conf": args.conf,
        "merge_sec": args.merge_sec,
        "padding_sec": args.padding_sec,
        "fast_cut": args.fast_cut,
        "verbose_timing": args.verbose_timing,
    }

    current_file_list = video_files
    method1_windows = {}

    for method in selected_methods:
        method_out_dir = os.path.join(output_dir, method_names[method])
        os.makedirs(method_out_dir, exist_ok=True)

        status = load_status(method_out_dir) if args.resume else {}
        to_process = [p for p in current_file_list if not (args.resume and p in status)]
        skipped = len(current_file_list) - len(to_process)
        if skipped:
            logging.info(f"Resume: skipping {skipped} already-processed files for {method_names[method]}")

        total_files = len(current_file_list)
        logging.info(f"Starting {method_names[method]} on {total_files} files "
                      f"({len(to_process)} to run, {skipped} resumed)")

        # Only restrict scanning range if Method 1 already ran earlier in THIS pipeline.
        tasks = []
        for path in to_process:
            windows = method1_windows.get(path) if (method != 1 and method1_windows) else None
            tasks.append((path, method_out_dir, roi_coords, method, windows, opts))

        next_file_list = []
        report_rows = []

        for path, info in status.items():
            if path in current_file_list:
                if info.get("hit"):
                    next_file_list.append(path)
                report_rows.append([path, info.get("hit"), info.get("start"), info.get("end"), info.get("error")])

        processed_count = skipped
        if tasks:
            with ProcessPoolExecutor(max_workers=args.workers,
                                      initializer=init_worker, initargs=(method,)) as executor:
                futures = {executor.submit(process_video_task, task): task for task in tasks}
                for future in as_completed(futures):
                    processed_count += 1
                    vid_path, hit_found, events, error = future.result()

                    status_entry = {"hit": hit_found, "error": error}
                    if events:
                        status_entry["start"] = events[0][0]
                        status_entry["end"] = events[-1][1]
                    status[vid_path] = status_entry

                    label = "HIT" if hit_found else ("ERROR" if error else "CLEAN")
                    logging.info(f"[{processed_count}/{total_files}] {os.path.basename(vid_path)} -> {label}"
                                  + (f" ({error})" if error else ""))

                    if hit_found:
                        next_file_list.append(vid_path)
                        if method == 1:
                            method1_windows[vid_path] = events

                    for s, e in (events or [(None, None)]):
                        report_rows.append([vid_path, hit_found, s, e, error])

        save_status(method_out_dir, status)
        write_report(method_out_dir, report_rows)

        current_file_list = sorted(set(next_file_list))
        if not current_file_list:
            logging.info("No hits passed to the next method. Pipeline stopped early.")
            break

    logging.info("All tasks completed")


if __name__ == "__main__":
    main()
