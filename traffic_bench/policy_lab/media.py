"""Operator-side GIF to MP4 conversion; these files never enter the sandbox."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def encoder_available() -> bool:
    if shutil.which("ffmpeg"):
        return True
    try:
        import cv2  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError:
        return False
    return True


def convert_gif(gif: Path) -> Path:
    """Produce a browser-friendly MP4 beside a GIF, retaining the original."""
    gif = Path(gif)
    target = gif.with_suffix(".mp4")
    ffmpeg_error = None
    if shutil.which("ffmpeg"):
        try:
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(gif), "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(target),
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if target.is_file() and target.stat().st_size > 0:
                return target
        except subprocess.CalledProcessError as exc:
            ffmpeg_error = exc.stderr.decode(errors="replace")[-500:] if exc.stderr else str(exc)

    try:
        import cv2
        import numpy as np
        from PIL import Image, ImageSequence
    except ImportError as exc:
        raise RuntimeError(f"ffmpeg conversion failed ({ffmpeg_error}); OpenCV/Pillow fallback unavailable") from exc

    with Image.open(gif) as source:
        width, height = source.size
        size = (width + width % 2, height + height % 2)
        fps = 25.0
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not open an MP4 writer; install ffmpeg with libx264")
        try:
            for frame in ImageSequence.Iterator(source):
                rgb = np.asarray(frame.convert("RGB"))
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if size != (width, height):
                    bgr = cv2.copyMakeBorder(bgr, 0, size[1] - height, 0, size[0] - width,
                                             cv2.BORDER_CONSTANT)
                repeats = max(1, round((frame.info.get("duration", 40) or 40) * fps / 1000))
                for _ in range(repeats):
                    writer.write(bgr)
        finally:
            writer.release()
    if not target.is_file() or target.stat().st_size == 0:
        raise RuntimeError(f"MP4 encoder produced no output for {gif}")
    return target


def convert_gifs(gif_dir: Path) -> tuple[int, list[str]]:
    count, errors = 0, []
    for gif in sorted(Path(gif_dir).glob("*.gif")):
        try:
            convert_gif(gif)
            count += 1
        except Exception as exc:
            errors.append(f"{gif.name}: {type(exc).__name__}: {exc}")
    return count, errors
