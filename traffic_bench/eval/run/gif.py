"""Top-down GIF film size / scaling, and GIF -> MP4 conversion.

    python -m traffic_bench.eval.run.gif DIR_OR_GIF... [--delete-gif] [--crf 18]

converts already written rollout GIFs to MP4 next to them.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from traffic_bench.eval.engine.sim.top_down_local_film_patch import set_top_down_local_film

# pygame segfaults around 24k; 4800 covers ~480 m at 10 px/m.
_SAFE_FILM_PX = 4800


def _topdown_gif_film_and_scaling(
    env,
    *,
    screen_size: tuple[int, int] = (800, 800),
    window_m: float = 80.0,
) -> tuple[tuple[int, int], float]:
    """Choose film_size + MetaDrive scaling for a follow-cam GIF.

    Dual-path crops can be kilometres across. We keep a 4800 px film and tell
    MetaDrive the map bbox is a box around the ego so zoom stays at ``window_m``.
    """
    screen = int(max(screen_size[0], screen_size[1]))
    win = float(window_m) if window_m and float(window_m) > 0.0 else 80.0
    scaling_req = float(screen) / win
    film = _SAFE_FILM_PX
    # MetaDrive: scaling = min(requested, film/max_len - 0.1)
    max_len = film / (scaling_req + 0.1) - 4.0
    half_m = max(win, 0.5 * max_len)
    vehicle = getattr(env, "vehicle", None)
    pos = getattr(vehicle, "position", None) if vehicle is not None else None
    if pos is not None:
        set_top_down_local_film((float(pos[0]), float(pos[1])), half_m=half_m)
    else:
        set_top_down_local_film(None)
    return (film, film), float(scaling_req)


_FFMPEG: str | None | bool = False  # False: not looked up yet


def find_ffmpeg() -> str | None:
    """System ffmpeg, else the binary bundled with imageio-ffmpeg (``pip install imageio-ffmpeg``)."""
    global _FFMPEG
    if _FFMPEG is False:
        _FFMPEG = shutil.which("ffmpeg")
        if _FFMPEG is None:
            try:
                import imageio_ffmpeg

                _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                _FFMPEG = None
    return _FFMPEG


def encode_mp4(frames, size: tuple[int, int], fps: float, mp4_path: Path, crf: int = 18) -> Path | None:
    """Stream PIL frames into H.264 MP4. Returns the MP4 path, or None without ffmpeg or on failure.

    The top-down view rotates with the ego, so every frame changes almost
    entirely and GIF's frame deltas do not help; H.264 at CRF 18 with
    ``-tune animation`` keeps the flat-colour frames visually identical at a
    small fraction of the size.
    """
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        return None
    width, height = size
    mp4_path = Path(mp4_path)
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", f"{fps:.6g}", "-i", "-",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264", "-preset", "medium", "-crf", str(int(crf)), "-tune", "animation",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(mp4_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in frames:
            if frame.size != (width, height):
                frame = frame.resize((width, height))
            proc.stdin.write(frame.convert("RGB").tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass
    err = proc.stderr.read().decode("utf-8", errors="replace")
    if proc.wait() != 0 or not mp4_path.is_file() or mp4_path.stat().st_size == 0:
        mp4_path.unlink(missing_ok=True)
        print(f"[gif] ffmpeg failed on {mp4_path.name}: {err.strip()[:500]}", file=sys.stderr, flush=True)
        return None
    return mp4_path


def renderer_to_mp4(renderer, mp4_path: Path, duration_ms: int = 40, crf: int = 18) -> Path | None:
    """MP4 of a MetaDrive top-down recording, without writing a GIF.

    MetaDrive's ``generate_gif`` turns its recorded frames into PIL images and
    hands them to ``Image.save``; that save (the slow, large GIF encode) is
    intercepted, so the MP4 gets exactly the frames, order and orientation the
    GIF would have had, whatever the MetaDrive version keeps internally.
    """
    from PIL import Image

    if find_ffmpeg() is None:
        return None
    captured: list = []
    real_save = Image.Image.save

    def capture(self, fp, format=None, **params):
        captured[:] = [self, *params.get("append_images", [])]

    Image.Image.save = capture
    try:
        renderer.generate_gif(str(Path(mp4_path).with_suffix(".gif")), duration=duration_ms)
    finally:
        Image.Image.save = real_save
    if not captured:
        return None
    return encode_mp4(captured, captured[0].size, 1000.0 / duration_ms, Path(mp4_path), crf)


def gif_to_mp4(gif_path: Path, mp4_path: Path | None = None, crf: int = 18) -> Path | None:
    """Re-encode an existing rollout GIF as MP4 at the same frame rate (frames are streamed)."""
    from PIL import Image, ImageSequence

    gif_path = Path(gif_path)
    with Image.open(gif_path) as im:
        duration_ms = float(im.info.get("duration") or 40) or 40.0
        return encode_mp4(ImageSequence.Iterator(im), im.size, 1000.0 / duration_ms,
                          Path(mp4_path) if mp4_path is not None else gif_path.with_suffix(".mp4"), crf)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Convert rollout GIFs to MP4 (H.264).")
    ap.add_argument("paths", nargs="+", type=Path, help="GIF files or directories of GIFs")
    ap.add_argument("--crf", type=int, default=18, help="H.264 quality: lower is better, 18 is visually lossless")
    ap.add_argument("--delete-gif", action="store_true", help="remove each GIF once its MP4 is written")
    args = ap.parse_args(argv)
    if find_ffmpeg() is None:
        raise SystemExit("no ffmpeg found: install it or `pip install imageio-ffmpeg`")
    gifs = [g for p in args.paths for g in (sorted(p.glob("*.gif")) if p.is_dir() else [p])]
    before = after = 0
    for gif in gifs:
        mp4 = gif_to_mp4(gif, crf=args.crf)
        if mp4 is None:
            continue
        before += gif.stat().st_size
        after += mp4.stat().st_size
        if args.delete_gif:
            gif.unlink()
    print(f"{len(gifs)} GIFs: {before / 1e6:.1f} MB -> {after / 1e6:.1f} MB MP4")


if __name__ == "__main__":
    main()
