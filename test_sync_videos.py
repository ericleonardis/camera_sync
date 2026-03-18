"""
Test suite for sync_videos.py

Generates short synthetic videos with controllable flash events,
then verifies the full pipeline: metadata reading, brightness extraction,
peak detection, alignment, plotting, JSON output, and ffmpeg rendering.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from sync_videos import (
    build_parser,
    detect_light_events,
    extract_frame_means,
    get_best_encoder,
    get_video_info,
    render_aligned_video,
    render_videos_parallel,
    run_sync,
    save_alignment_plot,
)

# ---------------------------------------------------------------------------
# Helpers — synthetic video generation
# ---------------------------------------------------------------------------

FPS = 30
WIDTH, HEIGHT = 64, 48  # tiny resolution for speed


def make_synthetic_video(
    path: str,
    duration_seconds: float = 3.0,
    fps: int = FPS,
    flash_frames: list[int] | None = None,
    base_brightness: int = 40,
    flash_brightness: int = 220,
) -> str:
    """Write a tiny .mp4 with optional bright-flash frames.

    Returns the path written to.
    """
    n_frames = int(duration_seconds * fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")

    flash_set = set(flash_frames or [])
    for i in range(n_frames):
        brightness = flash_brightness if i in flash_set else base_brightness
        frame = np.full((HEIGHT, WIDTH, 3), brightness, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture()
def tmp_videos(tmp_path):
    """Create 3 synthetic videos with flashes at known offsets.

    Video 0: flash at frame 20
    Video 1: flash at frame 30  (10-frame shift)
    Video 2: flash at frame 25  ( 5-frame shift)
    """
    paths = []
    flash_offsets = [20, 30, 25]
    for i, offset in enumerate(flash_offsets):
        p = str(tmp_path / f"cam{i}.mp4")
        make_synthetic_video(p, duration_seconds=3.0, flash_frames=[offset])
        paths.append(p)
    return paths, flash_offsets


@pytest.fixture()
def tmp_video_dir(tmp_path):
    """Create a directory with 2 synthetic videos for --input-dir tests."""
    d = tmp_path / "vids"
    d.mkdir()
    offsets = [15, 25]
    for i, offset in enumerate(offsets):
        make_synthetic_video(str(d / f"vid{i}.mp4"), flash_frames=[offset])
    return d, offsets


# ---------------------------------------------------------------------------
# Tests — video info
# ---------------------------------------------------------------------------

class TestGetVideoInfo:
    def test_reads_fps_and_duration(self, tmp_path):
        p = make_synthetic_video(str(tmp_path / "test.mp4"), duration_seconds=2.0)
        info = get_video_info(p)
        assert abs(info["fps"] - FPS) < 1
        assert info["frame_count"] == int(2.0 * FPS)
        assert abs(info["duration"] - 2.0) < 0.1

    def test_raises_on_missing_file(self):
        with pytest.raises(RuntimeError, match="Cannot open"):
            get_video_info("/nonexistent/video.mp4")


# ---------------------------------------------------------------------------
# Tests — brightness extraction
# ---------------------------------------------------------------------------

class TestExtractFrameMeans:
    def test_returns_correct_length(self, tmp_path):
        p = make_synthetic_video(str(tmp_path / "test.mp4"), duration_seconds=1.0)
        means = extract_frame_means(p, start_frame=0, end_frame=FPS)
        assert len(means) == FPS

    def test_detects_bright_frame(self, tmp_path):
        flash_frame = 10
        p = make_synthetic_video(
            str(tmp_path / "test.mp4"),
            duration_seconds=1.0,
            flash_frames=[flash_frame],
        )
        means = extract_frame_means(p, start_frame=0, end_frame=FPS)
        # The flash frame should be significantly brighter
        assert means[flash_frame] > means[0] + 50

    def test_partial_range(self, tmp_path):
        p = make_synthetic_video(str(tmp_path / "test.mp4"), duration_seconds=2.0)
        means = extract_frame_means(p, start_frame=10, end_frame=20)
        assert len(means) == 10


# ---------------------------------------------------------------------------
# Tests — peak detection
# ---------------------------------------------------------------------------

class TestDetectLightEvents:
    def test_finds_single_peak(self):
        signal = np.full(90, 40.0)
        signal[30] = 220.0
        peaks = detect_light_events(signal, top_k=3)
        assert 30 in peaks

    def test_finds_multiple_peaks(self):
        signal = np.full(200, 40.0)
        for pos in [30, 80, 150]:
            signal[pos] = 220.0
        peaks = detect_light_events(signal, top_k=5)
        assert len(peaks) >= 3
        for pos in [30, 80, 150]:
            assert any(abs(p - pos) <= 2 for p in peaks)

    def test_top_k_limits_peaks(self):
        signal = np.full(300, 40.0)
        for pos in range(20, 280, 20):
            signal[pos] = 220.0
        peaks = detect_light_events(signal, top_k=3)
        assert len(peaks) <= 3

    def test_returns_sorted(self):
        signal = np.full(200, 40.0)
        signal[100] = 250.0
        signal[30] = 230.0
        signal[160] = 240.0
        peaks = detect_light_events(signal, top_k=5)
        assert list(peaks) == sorted(peaks)


# ---------------------------------------------------------------------------
# Tests — encoder selection
# ---------------------------------------------------------------------------

class TestGetBestEncoder:
    def test_default_is_libx264(self):
        enc, flags = get_best_encoder(None)
        assert enc == "libx264"
        assert "-preset" in flags
        assert "superfast" in flags

    def test_preferred_nonsense_falls_back(self):
        enc, flags = get_best_encoder("totally_fake_encoder_xyz")
        assert enc == "libx264"


# ---------------------------------------------------------------------------
# Tests — render single video
# ---------------------------------------------------------------------------

class TestRenderAlignedVideo:
    def test_renders_without_shift(self, tmp_path):
        src = make_synthetic_video(str(tmp_path / "src.mp4"), duration_seconds=1.0)
        out = str(tmp_path / "out.mp4")
        result = render_aligned_video(
            src, out, shift_seconds=0, duration=None,
            encoder="libx264",
            encoder_flags=["-preset", "superfast", "-crf", "23"],
        )
        assert result["returncode"] == 0
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0

    def test_renders_with_shift(self, tmp_path):
        src = make_synthetic_video(str(tmp_path / "src.mp4"), duration_seconds=2.0)
        out = str(tmp_path / "out.mp4")
        result = render_aligned_video(
            src, out, shift_seconds=0.5, duration=1.0,
            encoder="libx264",
            encoder_flags=["-preset", "superfast", "-crf", "23"],
        )
        assert result["returncode"] == 0
        info = get_video_info(out)
        # Should be roughly 1 second
        assert abs(info["duration"] - 1.0) < 0.5

    def test_renders_with_duration_limit(self, tmp_path):
        src = make_synthetic_video(str(tmp_path / "src.mp4"), duration_seconds=3.0)
        out = str(tmp_path / "out.mp4")
        result = render_aligned_video(
            src, out, shift_seconds=0, duration=1.0,
            encoder="libx264",
            encoder_flags=["-preset", "superfast", "-crf", "23"],
        )
        assert result["returncode"] == 0
        info = get_video_info(out)
        assert info["duration"] < 2.0


# ---------------------------------------------------------------------------
# Tests — parallel rendering
# ---------------------------------------------------------------------------

class TestRenderVideosParallel:
    def test_renders_multiple(self, tmp_path):
        vids = []
        for i in range(3):
            vids.append(make_synthetic_video(
                str(tmp_path / f"cam{i}.mp4"), duration_seconds=1.5,
            ))
        out_dir = tmp_path / "output"
        results = render_videos_parallel(
            vids,
            shifts_seconds=[0.0, 0.1, 0.2],
            output_dir=out_dir,
            encoder="libx264",
            encoder_flags=["-preset", "superfast", "-crf", "23"],
            max_workers=2,
        )
        assert len(results) == 3
        assert all(r["returncode"] == 0 for r in results)
        assert len(list(out_dir.glob("*_aligned.mp4"))) == 3


# ---------------------------------------------------------------------------
# Tests — plotting
# ---------------------------------------------------------------------------

class TestSaveAlignmentPlot:
    def test_saves_png(self, tmp_path):
        signals = [np.random.rand(100) for _ in range(3)]
        peaks = [np.array([20, 50]), np.array([25, 55]), np.array([22, 52])]
        aligned = [s[10:] for s in signals]
        plot_path = tmp_path / "plot.png"
        save_alignment_plot(signals, peaks, aligned, ["a", "b", "c"], plot_path)
        assert plot_path.exists()
        assert plot_path.stat().st_size > 1000


# ---------------------------------------------------------------------------
# Tests — full pipeline via run_sync (integration)
# ---------------------------------------------------------------------------

class TestRunSyncIntegration:
    def _make_args(self, **kwargs):
        """Build an argparse Namespace with defaults + overrides."""
        parser = build_parser()
        defaults = parser.parse_args([])
        for k, v in kwargs.items():
            setattr(defaults, k, v)
        return defaults

    def test_analyze_only_with_positional_videos(self, tmp_videos, tmp_path):
        paths, offsets = tmp_videos
        out = tmp_path / "results"
        args = self._make_args(
            videos=paths,
            output_dir=str(out),
            analyze_only=True,
            top_k=1,
        )
        run_sync(args)

        # Check outputs
        assert (out / "alignment_plot.png").exists()
        assert (out / "camera_shifts.json").exists()
        assert (out / "camera_shifts.npy").exists()

        with open(out / "camera_shifts.json") as f:
            data = json.load(f)

        assert len(data["shifts_frames"]) == 3
        assert data["fps"] == FPS
        # Video 0 has earliest flash (frame 20), so its shift should be 0
        assert data["shifts_frames"][0] == 0
        # Video 1 flash at 30 => shift = 10
        assert data["shifts_frames"][1] == 10
        # Video 2 flash at 25 => shift = 5
        assert data["shifts_frames"][2] == 5

        # No rendered videos since analyze_only
        assert len(list(out.glob("*_aligned.mp4"))) == 0

    def test_full_pipeline_with_rendering(self, tmp_videos, tmp_path):
        paths, _ = tmp_videos
        out = tmp_path / "rendered"
        args = self._make_args(
            videos=paths,
            output_dir=str(out),
            analyze_only=False,
            top_k=1,
        )
        run_sync(args)

        assert (out / "camera_shifts.json").exists()
        aligned_vids = list(out.glob("*_aligned.mp4"))
        assert len(aligned_vids) == 3
        for v in aligned_vids:
            assert v.stat().st_size > 0

    def test_input_dir_mode(self, tmp_video_dir, tmp_path):
        vid_dir, offsets = tmp_video_dir
        out = tmp_path / "dir_results"
        args = self._make_args(
            input_dir=str(vid_dir),
            output_dir=str(out),
            analyze_only=True,
            top_k=1,
        )
        run_sync(args)

        with open(out / "camera_shifts.json") as f:
            data = json.load(f)
        assert len(data["shifts_frames"]) == 2
        # vid0 flash at 15, vid1 at 25 => shifts 0 and 10
        assert data["shifts_frames"][0] == 0
        assert data["shifts_frames"][1] == 10

    def test_match_duration(self, tmp_videos, tmp_path):
        paths, _ = tmp_videos
        out = tmp_path / "matched"
        args = self._make_args(
            videos=paths,
            output_dir=str(out),
            analyze_only=False,
            match_duration=True,
            top_k=1,
        )
        run_sync(args)

        aligned_vids = list(out.glob("*_aligned.mp4"))
        assert len(aligned_vids) == 3
        # All outputs should exist and have similar durations
        durations = []
        for v in aligned_vids:
            info = get_video_info(str(v))
            durations.append(info["duration"])
        # Durations should be within 0.5s of each other
        assert max(durations) - min(durations) < 0.5

    def test_fps_override(self, tmp_videos, tmp_path):
        paths, _ = tmp_videos
        out = tmp_path / "fps_override"
        args = self._make_args(
            videos=paths,
            output_dir=str(out),
            fps=30.0,
            analyze_only=True,
            top_k=1,
        )
        run_sync(args)

        with open(out / "camera_shifts.json") as f:
            data = json.load(f)
        assert data["fps"] == 30.0


# ---------------------------------------------------------------------------
# Tests — CLI argument parsing
# ---------------------------------------------------------------------------

class TestCLIParsing:
    def test_defaults(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.fps is None
        assert args.start_second is None
        assert args.end_second is None
        assert args.top_k == 7
        assert args.encoder is None
        assert args.output_dir == "./output"
        assert args.analyze_only is False
        assert args.match_duration is False

    def test_positional_videos(self):
        parser = build_parser()
        args = parser.parse_args(["a.mp4", "b.mp4"])
        assert args.videos == ["a.mp4", "b.mp4"]

    def test_all_flags(self):
        parser = build_parser()
        args = parser.parse_args([
            "-d", "/some/dir",
            "-o", "/out",
            "--fps", "24",
            "--start-second", "5",
            "--end-second", "60",
            "--top-k", "3",
            "--encoder", "hevc_nvenc",
            "--duration", "30",
            "--workers", "4",
            "--analyze-only",
        ])
        assert args.input_dir == "/some/dir"
        assert args.output_dir == "/out"
        assert args.fps == 24.0
        assert args.start_second == 5.0
        assert args.end_second == 60.0
        assert args.top_k == 3
        assert args.encoder == "hevc_nvenc"
        assert args.duration == 30.0
        assert args.workers == 4
        assert args.analyze_only is True
