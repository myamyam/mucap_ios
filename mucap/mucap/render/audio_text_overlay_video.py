"""Shared compositor for placing an audio-text render over a performance video."""

from __future__ import annotations

import subprocess
from pathlib import Path


OUTPUT_SIZE = (1280, 720)
DEFAULT_PANEL_HEIGHT = 360
BACKGROUND_OPACITY = 0.5


def find_source_video(wav_path_or_name: str | Path) -> Path | None:
    """Find an MP4 in its instrument folder, then fall back to recursive search."""
    stem = Path(wav_path_or_name).stem
    instrument = stem.split("_", 1)[0]
    project_root = Path(__file__).resolve().parents[2]
    if stem.casefold().startswith("newoboe_"):
        original_stem = stem[len("newoboe_"):]
        new_oboe_match = (
            project_root / "input" / "input_video" / "newoboe"
            / f"oboe_{original_stem}.mp4"
        )
        if new_oboe_match.is_file():
            return new_oboe_match
    if stem.casefold().startswith("vid_"):
        new_vid_match = project_root / "input" / "new_vid" / f"{stem}.mp4"
        if new_vid_match.is_file():
            return new_vid_match
    if stem.casefold().startswith("mov_"):
        mov_vid_match = project_root / "input" / "mov_vid" / f"{stem}.mp4"
        if mov_vid_match.is_file():
            return mov_vid_match
    input_vid_match = project_root / "input" / "input_vid" / f"{stem}.mp4"
    if input_vid_match.is_file():
        return input_vid_match
    video_root = project_root / "input" / "input_video"
    instrument_match = video_root / instrument / f"{stem}.mp4"
    if instrument_match.is_file():
        return instrument_match

    matches = sorted(
        path for path in video_root.rglob("*.mp4")
        if path.stem.casefold() == stem.casefold()
    )
    if not matches:
        return None
    if len(matches) > 1:
        print(
            f"Warning: multiple source videos found for {stem}; "
            f"using {matches[0]}"
        )
    return matches[0]


def compose_audio_text_overlay(
    source_video: str | Path,
    text_video: str | Path,
    output_path: str | Path,
    panel_height: int = DEFAULT_PANEL_HEIGHT,
    bottom_offset: int = 0,
    panel_width: int | None = None,
) -> Path:
    """Bottom-align a translucent white text panel over the source video."""
    import imageio_ffmpeg

    source_video = Path(source_video).resolve()
    text_video = Path(text_video).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = OUTPUT_SIZE
    panel_width = panel_width or width
    panel_x = (width - panel_width) // 2
    panel_y = height - panel_height - bottom_offset
    filter_complex = (
        f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height}[base];"
        f"[1:v]scale={panel_width}:{panel_height}[panel];"
        f"[base]drawbox=x={panel_x}:y={panel_y}:"
        f"w={panel_width}:h={panel_height}:"
        f"color=white@{BACKGROUND_OPACITY}:t=fill[washed];"
        # A blended key keeps anti-aliased glyph edges from becoming an opaque
        # white halo. It also preserves karaoke's pending text as genuinely
        # faint instead of treating its near-white pixels as solid lettering.
        "[panel]format=rgba,colorkey=0xFFFFFF:0.02:0.20[text];"
        f"[washed][text]overlay={panel_x}:H-h-{bottom_offset}:shortest=1[v]"
    )

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-i",
        str(source_video),
        "-i",
        str(text_video),
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    return output_path
