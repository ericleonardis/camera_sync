#!/usr/bin/env python3
"""
Multi-Camera Video Synchronization CLI Tool

Detects flash events across multiple camera recordings, computes alignment
offsets, and renders synchronized output videos with parallel processing.

FPS and analysis window are read from the video files by default.
Default encoder: libx264 -preset superfast -crf 23.
Pass --encoder h264_nvenc to use GPU-accelerated NVENC encoding.

Usage:
    # Synchronize specific video files (auto-detects FPS, scans full video)
    python sync_videos.py video1.mp4 video2.mp4 video3.mp4

    # Synchronize all videos in a folder
    python sync_videos.py --input-dir /path/to/videos/

    # Analyze only (no rendering)
    python sync_videos.py --input-dir /path/to/videos/ --analyze-only

    # GPU-accelerated encoding
    python sync_videos.py --input-dir /path/to/videos/ --encoder h264_nvenc

    # Override FPS and analysis window
    python sync_videos.py --input-dir /path/to/videos/ --fps 30 --start-second 10 --end-second 60
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def get_video_info(video_path: str) -> dict:
    """Read FPS, frame count, and duration from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return {"fps": fps, "frame_count": frame_count, "duration": duration}


def extract_frame_means(video_path: str, start_frame: int, end_frame: int) -> np.ndarray:
    """Extract per-frame mean pixel brightness from a video segment."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    means = []
    n_frames = end_frame - start_frame
    for _ in tqdm(range(n_frames), desc=f"  {Path(video_path).name}", leave=False):
        ret, frame = cap.read()
        if not ret:
            break
        means.append(np.mean(frame))
    cap.release()
    return np.array(means)


def detect_light_events(signal_data: np.ndarray, distance: int = 10, top_k: int = 7) -> np.ndarray:
    """Detect the strongest light flash events via peak detection."""
    signal_mean = np.mean(signal_data)
    signal_range = np.max(signal_data) - np.min(signal_data)
    auto_prominence = max(20, signal_range * 0.15)

    peaks, props = find_peaks(
        signal_data,
        prominence=auto_prominence,
        distance=distance,
        height=signal_mean + auto_prominence * 0.4,
    )

    if len(peaks) < top_k:
        peaks, props = find_peaks(
            signal_data,
            prominence=auto_prominence * 0.6,
            distance=distance,
            height=signal_mean + auto_prominence * 0.3,
        )

    if len(peaks) > top_k:
        sort_idx = np.argsort(props["peak_heights"])[::-1]
        peaks = peaks[sort_idx[:top_k]]

    return np.sort(peaks)


# ---------------------------------------------------------------------------
# GPU-accelerated rendering via ffmpeg + NVENC
# ---------------------------------------------------------------------------

def probe_encoder(encoder: str) -> bool:
    """Check if an ffmpeg encoder is available."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        return encoder in r.stdout
    except Exception:
        return False


def get_best_encoder(preferred: str | None) -> tuple[str, list[str]]:
    """Return (encoder_name, extra_flags) for the best available encoder.

    Default: libx264 -preset superfast -crf 23 (matches standard ffmpeg workflow).
    Pass --encoder h264_nvenc (or similar) to use GPU encoding instead.
    """
    if preferred:
        gpu_candidates = {
            "h264_nvenc": ["-preset", "p4", "-rc", "vbr", "-cq", "23", "-gpu", "any"],
            "hevc_nvenc": ["-preset", "p4", "-rc", "vbr", "-cq", "23", "-gpu", "any"],
            "av1_nvenc":  ["-preset", "p4", "-rc", "vbr", "-cq", "23", "-gpu", "any"],
            "h264_vaapi": ["-vaapi_device", "/dev/dri/renderD128"],
        }
        if preferred in gpu_candidates and probe_encoder(preferred):
            return preferred, gpu_candidates[preferred]
        if probe_encoder(preferred):
            return preferred, ["-preset", "superfast", "-crf", "23"]

    # Default: libx264 superfast (universally available, good quality/speed)
    return "libx264", ["-preset", "superfast", "-crf", "23"]


def render_aligned_video(
    input_path: str,
    output_path: str,
    shift_seconds: float,
    duration: float | None,
    encoder: str,
    encoder_flags: list[str],
    gpu_id: int = 0,
) -> dict:
    """Render a single aligned video using ffmpeg with GPU acceleration."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning"]

    # Use CUDA for decoding if using NVENC
    if "nvenc" in encoder:
        cmd += ["-hwaccel", "cuda", "-hwaccel_device", str(gpu_id)]

    # Input with seek
    if shift_seconds > 0:
        cmd += ["-ss", f"{shift_seconds:.6f}"]
    cmd += ["-i", input_path]

    # Duration limit
    if duration is not None:
        cmd += ["-t", f"{duration:.6f}"]

    # Encoding
    cmd += ["-c:v", encoder] + encoder_flags
    cmd += ["-pix_fmt", "yuv420p"]

    # Audio
    cmd += ["-c:a", "aac", "-b:a", "128k"]

    cmd.append(output_path)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    return {
        "input": input_path,
        "output": output_path,
        "returncode": result.returncode,
        "stderr": result.stderr.strip(),
        "cmd": " ".join(cmd),
    }


def render_videos_parallel(
    video_paths: list[str],
    shifts_seconds: list[float],
    output_dir: Path,
    encoder: str,
    encoder_flags: list[str],
    duration: float | None = None,
    max_workers: int | None = None,
) -> list[dict]:
    """Render all aligned videos in parallel using GPU-accelerated encoding."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect number of GPUs for round-robin assignment
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        n_gpus = max(1, len(r.stdout.strip().splitlines()))
    except Exception:
        n_gpus = 1

    if max_workers is None:
        max_workers = min(len(video_paths), n_gpus * 2)

    tasks = []
    for i, (vpath, shift_s) in enumerate(zip(video_paths, shifts_seconds)):
        stem = Path(vpath).stem
        out = str(output_dir / f"{stem}_aligned.mp4")
        gpu_id = i % n_gpus
        tasks.append((vpath, out, shift_s, duration, encoder, encoder_flags, gpu_id))

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(render_aligned_video, *t): t[0] for t in tasks}
        for future in as_completed(futures):
            res = future.result()
            name = Path(res["input"]).name
            if res["returncode"] == 0:
                print(f"  [OK]   {name} -> {Path(res['output']).name}")
            else:
                print(f"  [FAIL] {name}: {res['stderr'][:200]}")
            results.append(res)

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_alignment_plot(
    frame_means: list[np.ndarray],
    all_peaks: list[np.ndarray],
    aligned_segments: list[np.ndarray],
    video_names: list[str],
    output_path: Path,
):
    """Save before/after alignment plot to disk."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=False)

    for i, (signal, peaks) in enumerate(zip(frame_means, all_peaks)):
        axes[0].plot(signal, label=video_names[i])
        if len(peaks) > 0:
            axes[0].plot(peaks, signal[peaks], "x", color="red")
    axes[0].set_title("Before Alignment (Peaks Marked)")
    axes[0].set_ylabel("Mean Pixel Value")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for i, signal in enumerate(aligned_segments):
        axes[1].plot(signal, label=video_names[i])
    axes[1].set_title("After Alignment (First Peaks at t=0)")
    axes[1].set_ylabel("Mean Pixel Value")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    zoom = 500
    for i, signal in enumerate(aligned_segments):
        axes[2].plot(signal[:zoom], label=video_names[i])
    axes[2].set_title(f"Zoom: First {zoom} Frames After Alignment")
    axes[2].set_xlabel("Frame (aligned)")
    axes[2].set_ylabel("Mean Pixel Value")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Alignment plot saved to {output_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def resolve_video_paths(args) -> list[str]:
    """Resolve video file paths from CLI arguments."""
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg", ".wmv", ".m4v"}
    paths = []

    if args.input_dir:
        d = Path(args.input_dir)
        if not d.is_dir():
            sys.exit(f"Error: {args.input_dir} is not a directory")
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in video_exts:
                paths.append(str(f))
        if not paths:
            sys.exit(f"Error: no video files found in {args.input_dir}")

    if args.videos:
        for v in args.videos:
            p = Path(v)
            if not p.is_file():
                sys.exit(f"Error: {v} not found")
            paths.append(str(p))

    if not paths:
        sys.exit("Error: provide video files or --input-dir")

    return paths


def run_sync(args):
    video_paths = resolve_video_paths(args)
    video_names = [Path(p).name for p in video_paths]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Read video metadata (FPS, duration) from the files ---
    video_infos = []
    for vp in video_paths:
        video_infos.append(get_video_info(vp))

    # Determine FPS: use --fps override, or read from the first video
    if args.fps is not None:
        fps = args.fps
    else:
        fps = video_infos[0]["fps"]
    print(f"Using FPS: {fps}")

    # Check for FPS mismatches across videos
    for i, info in enumerate(video_infos):
        if abs(info["fps"] - fps) > 1:
            print(f"  Warning: {video_names[i]} has FPS={info['fps']:.1f}, expected {fps:.1f}")

    # Determine analysis window: use overrides, or scan the full video
    start_second = args.start_second if args.start_second is not None else 0
    if args.end_second is not None:
        end_second = args.end_second
    else:
        # Use the shortest video duration so all videos have data
        end_second = min(info["duration"] for info in video_infos)

    print(f"Videos to synchronize ({len(video_paths)}):")
    for i, name in enumerate(video_names):
        info = video_infos[i]
        print(f"  - {name}  ({info['fps']:.1f} fps, {info['duration']:.1f}s, {info['frame_count']} frames)")

    # --- Step 1: Extract brightness signals ---
    start_frame = int(start_second * fps)
    end_frame = int(end_second * fps)
    print(f"\nExtracting brightness ({start_second:.1f}s–{end_second:.1f}s @ {fps} fps, "
          f"frames {start_frame}–{end_frame})...")
    frame_means = []
    for vp in video_paths:
        means = extract_frame_means(vp, start_frame, end_frame)
        frame_means.append(means)
        print(f"  {Path(vp).name}: {len(means)} frames")

    # --- Step 2: Detect flash peaks ---
    print(f"\nDetecting top {args.top_k} flash events...")
    all_peaks = []
    for i, signal in enumerate(frame_means):
        peaks = detect_light_events(signal, top_k=args.top_k)
        all_peaks.append(peaks)
        print(f"  {video_names[i]}: {len(peaks)} peaks at frames {peaks.tolist()}")

    # --- Step 3: Compute alignment shifts ---
    first_peaks = []
    for i, peaks in enumerate(all_peaks):
        if len(peaks) == 0:
            sys.exit(f"Error: no peaks detected in {video_names[i]}. "
                     "Try adjusting --start-second/--end-second or --top-k.")
        first_peaks.append(peaks[0])

    min_first = min(first_peaks)
    shifts_frames = [fp - min_first for fp in first_peaks]
    shifts_seconds = [s / fps for s in shifts_frames]

    print("\nAlignment results:")
    print(f"{'Video':<30} {'First Peak (frame)':<20} {'Shift (frames)':<16} {'Shift (seconds)'}")
    for i, name in enumerate(video_names):
        print(f"  {name:<28} {first_peaks[i]:<20} {shifts_frames[i]:<16} {shifts_seconds[i]:.4f}")

    # --- Step 4: Build aligned segments for plotting ---
    aligned_segments = []
    for shift, signal in zip(shifts_frames, frame_means):
        aligned = signal[shift:] if shift > 0 else signal.copy()
        aligned_segments.append(aligned)
    min_len = min(len(s) for s in aligned_segments)
    aligned_segments = [s[:min_len] for s in aligned_segments]

    # --- Step 5: Save outputs ---
    plot_path = output_dir / "alignment_plot.png"
    save_alignment_plot(frame_means, all_peaks, aligned_segments, video_names, plot_path)

    shift_data = {
        "video_paths": video_paths,
        "video_names": video_names,
        "first_peak_frames": [int(x) for x in first_peaks],
        "shifts_frames": [int(x) for x in shifts_frames],
        "shifts_seconds": shifts_seconds,
        "fps": fps,
        "method": "first_peak_alignment",
        "top_k": args.top_k,
    }
    json_path = output_dir / "camera_shifts.json"
    with open(json_path, "w") as f:
        json.dump(shift_data, f, indent=2)
    print(f"Shift data saved to {json_path}")

    npy_path = output_dir / "camera_shifts.npy"
    np.save(npy_path, shift_data)

    if args.analyze_only:
        print("\n--analyze-only set, skipping video rendering.")
        return

    # --- Step 6: Render aligned videos in parallel ---
    print(f"\nRendering aligned videos...")
    encoder, enc_flags = get_best_encoder(args.encoder)
    print(f"  Encoder: {encoder}")

    duration = None
    if args.duration:
        duration = args.duration
    elif args.match_duration:
        durations = []
        for info, shift_s in zip(video_infos, shifts_seconds):
            durations.append(info["duration"] - shift_s)
        duration = max(0, min(durations))
        print(f"  Common duration: {duration:.2f}s")

    results = render_videos_parallel(
        video_paths, shifts_seconds, output_dir,
        encoder, enc_flags,
        duration=duration,
        max_workers=args.workers,
    )

    ok = sum(1 for r in results if r["returncode"] == 0)
    fail = len(results) - ok
    print(f"\nDone: {ok} succeeded, {fail} failed.")
    if fail:
        for r in results:
            if r["returncode"] != 0:
                print(f"  FAILED: {r['input']}\n    {r['stderr'][:300]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Synchronize multi-camera videos using flash detection and render aligned outputs with GPU acceleration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("videos", nargs="*", help="Video file paths to synchronize")
    p.add_argument("-d", "--input-dir", help="Directory containing video files")
    p.add_argument("-o", "--output-dir", default="./output", help="Output directory (default: ./output)")

    # Analysis parameters
    analysis = p.add_argument_group("analysis")
    analysis.add_argument("--fps", type=float, default=None, help="Frame rate override (default: read from video)")
    analysis.add_argument("--start-second", type=float, default=None, help="Start time for brightness analysis (default: 0)")
    analysis.add_argument("--end-second", type=float, default=None, help="End time for brightness analysis (default: full video)")
    analysis.add_argument("--top-k", type=int, default=7, help="Number of flash peaks to detect (default: 7)")

    # Rendering parameters
    render = p.add_argument_group("rendering")
    render.add_argument("--encoder", help="FFmpeg encoder (default: libx264; use h264_nvenc for GPU)")
    render.add_argument("--duration", type=float, help="Output duration in seconds")
    render.add_argument("--match-duration", action="store_true", help="Trim all outputs to the shortest common duration")
    render.add_argument("--workers", type=int, help="Max parallel render workers (default: auto)")
    render.add_argument("--analyze-only", action="store_true", help="Only analyze alignment, skip rendering")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_sync(args)


if __name__ == "__main__":
    main()
