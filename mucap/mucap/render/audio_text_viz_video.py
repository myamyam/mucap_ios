"""Render audio_text_viz captions to MP4.

The default ``direct`` mode uses the shared visual calculations from
``audio_text_viz.py`` for smoother video. The optional ``html`` mode creates or
loads the HTML produced by ``visualize_captions()`` and captures that exact
browser-rendered result into an MP4.
"""

from __future__ import annotations

import argparse
import colorsys
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from mucap.render.audio_text_viz import (
    CAPTION_FONT_SCALE,
    CAPTION_MAPPING,
    DEFAULT_FEATURES_CSV,
    DEFAULT_NOTES_CSV,
    DEFAULT_OUT_DIR,
    INTERNAL_COLOR_NORM_DELTA,
    INTERNAL_OPACITY_NORM_DELTA,
    INTERNAL_OPACITY_RANGE,
    INTERNAL_PITCH_OFFSET_PX,
    _audio_key,
    _caption_end_time,
    _duration_from_captions,
    _opacity_style,
    _palette_color,
    _pitch_norm,
    _pitch_offset_px,
    _row_at_time,
    _visual_mode_from_context,
    apply_note_mapping,
    load_song,
    note_table,
    note_index_at,
    parse_captions,
    visualize_captions,
)

DEFAULT_VIDEO_FPS = 30
DEFAULT_VIDEO_SIZE = (1280, 720)


def _hsl_to_rgb(hue: float, sat: float, lightness: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hls_to_rgb((hue % 360.0) / 360.0, lightness / 100.0, sat / 100.0)
    return int(r * 255), int(g * 255), int(b * 255)


def _file_url(path: Path) -> str:
    return path.resolve().as_uri()


def _require_playwright():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "audio_text_viz_video requires Playwright to capture the HTML. "
            "Install it in the venv with: pip install playwright"
        ) from exc
    return sync_playwright


def _duration_for_video(captions: object, html_path: Path) -> float:
    try:
        parsed = parse_captions(captions)
        file_name = _audio_key(html_path.stem.replace("-audio_text_viz", ""))
        notes, _ = load_song(file_name)
        return _duration_from_captions(parsed, note_table(notes))
    except Exception:
        return 1.0


def _pad_to_macroblock(frame: np.ndarray, block_size: int = 16) -> np.ndarray:
    """Pad, never scale, so browser pixels stay intact in H.264-friendly video."""
    if block_size <= 1:
        return frame
    height, width = frame.shape[:2]
    padded_width = int(math.ceil(width / block_size) * block_size)
    padded_height = int(math.ceil(height / block_size) * block_size)
    if padded_width == width and padded_height == height:
        return frame

    out = np.full((padded_height, padded_width, frame.shape[2]), 255, dtype=frame.dtype)
    out[:height, :width, :] = frame
    return out


def _rgb_for_row(row, df, visual_mode: str, base_norm: float | None = None) -> tuple[int, int, int]:
    norm = _pitch_norm(row, df)
    if base_norm is not None:
        norm = float(
            np.clip(
                base_norm + np.clip(
                    norm - base_norm,
                    -INTERNAL_COLOR_NORM_DELTA,
                    INTERNAL_COLOR_NORM_DELTA,
                ),
                0.0,
                1.0,
            )
        )
    hue, sat, lightness = _palette_color(norm, visual_mode)
    return _hsl_to_rgb(hue, sat, lightness)


def _font_for_size(size_px: float):
    from PIL import ImageFont

    font_path = Path(__file__).resolve().parents[1] / "fonts" / "Pretendard-Bold.otf"
    size = max(12, int(round(size_px)))
    if font_path.exists():
        return ImageFont.truetype(str(font_path), size)
    return ImageFont.load_default()


def _text_size(draw, text: str, font) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _view_start(current: float, duration: float, window_size: float = 7.0) -> float:
    visible = min(window_size, duration)
    if duration <= visible:
        return 0.0
    if current < visible / 2:
        return 0.0
    if current > duration - visible / 2:
        return duration - visible
    return current - visible / 2


def _draw_grid(draw, width: int, height: int) -> None:
    stage_height = height - 34
    for i in range(9):
        x = int(width * i / 8)
        draw.line((x, 0, x, stage_height), fill=(224, 224, 224), width=1)
    draw.line((0, stage_height // 2, width, stage_height // 2), fill=(235, 235, 235), width=1)
    draw.line((0, stage_height, width, stage_height), fill=(80, 80, 80), width=1)


def _draw_token_direct(img, draw, token: str, x: float, y: float, row,
                       timestamp: float, end_time: float, df, visual_mode: str,
                       current: float) -> None:
    from PIL import Image, ImageDraw

    chars = list(token)
    if not chars:
        return

    font = _font_for_size(float(row["size_px"]) * CAPTION_FONT_SCALE)
    widths = [_text_size(draw, ch, font)[0] for ch in chars]
    total_width = sum(widths)
    base_norm = _pitch_norm(row, df)
    token_offset_y = _pitch_offset_px(row, df)
    pop_age = current - timestamp
    pop_y = -18.0 if 0.0 <= pop_age <= 0.32 else 0.0
    cursor_x = x - total_width / 2.0

    for j, ch in enumerate(chars):
        char_time = timestamp + ((j + 0.5) / len(chars)) * max(end_time - timestamp, 0.05)
        char_row = _row_at_time(char_time, df)
        if char_row is None:
            char_row = row
        raw_offset_y = _pitch_offset_px(char_row, df)
        offset_y = float(
            np.clip(raw_offset_y - token_offset_y, -INTERNAL_PITCH_OFFSET_PX, INTERNAL_PITCH_OFFSET_PX)
        )
        opacity = _opacity_style(char_row, df, base_norm)
        rgb = _rgb_for_row(char_row, df, visual_mode, base_norm)
        alpha = int(np.clip(opacity, 0.0, 1.0) * 255)
        layer = Image.new("RGBA", img.size, (255, 255, 255, 0))
        layer_draw = ImageDraw.Draw(layer)
        layer_draw.text((cursor_x, y + offset_y + pop_y), ch, font=font, fill=(*rgb, alpha))
        img.alpha_composite(layer)
        cursor_x += widths[j]


def _direct_video(
    wav_path_or_name: str,
    captions: object,
    out_path: str | Path,
    notes_csv: Path = DEFAULT_NOTES_CSV,
    features_csv: Path | None = DEFAULT_FEATURES_CSV,
    feature_row: dict | None = None,
    audio_path: str | Path | None = None,
    fps: int = DEFAULT_VIDEO_FPS,
    size: tuple[int, int] = DEFAULT_VIDEO_SIZE,
) -> Path:
    """Smooth MP4 renderer using audio_text_viz's shared visual calculations."""
    import imageio
    from PIL import Image, ImageDraw

    from mucap.render.utils import add_audio_to_video

    file_name = _audio_key(wav_path_or_name)
    notes, song_row = load_song(file_name, notes_csv, features_csv, feature_row)
    df = apply_note_mapping(note_table(notes), CAPTION_MAPPING)
    context_row = dict(song_row)
    if feature_row:
        context_row.update(feature_row)
    visual_mode = _visual_mode_from_context(audio_path or wav_path_or_name, context_row)
    parsed_captions = parse_captions(captions)
    duration = _duration_from_captions(parsed_captions, df)
    width, height = size
    stage_height = height - 34

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

    try:
        writer = imageio.get_writer(temp_video, fps=fps, codec="libx264", quality=9, macro_block_size=16)
        total_frames = max(1, int(math.ceil(duration * fps)))
        for frame_index in range(total_frames):
            current = frame_index / float(fps)
            start = _view_start(current, duration)
            visible = min(7.0, duration)
            end = start + visible
            img = Image.new("RGBA", (width, height), (255, 255, 255, 255))
            draw = ImageDraw.Draw(img)
            _draw_grid(draw, width, height)

            for index, cap in enumerate(parsed_captions):
                timestamp = float(cap["time"])
                if timestamp < start - 0.001 or timestamp > end + 0.001:
                    continue
                if df.empty:
                    continue
                row = df.iloc[note_index_at(timestamp, df)]
                x = max(0.0, min(1.0, (timestamp - start) / visible)) * width
                y = stage_height / 2.0 + _pitch_offset_px(row, df)
                caption_end = _caption_end_time(index, parsed_captions, duration)
                _draw_token_direct(
                    img,
                    draw,
                    str(cap["text"]),
                    x,
                    y,
                    row,
                    timestamp,
                    caption_end,
                    df,
                    visual_mode,
                    current,
                )

            cursor_x = max(0.0, min(1.0, (current - start) / visible)) * width
            draw.rectangle((cursor_x - 1, 0, cursor_x + 1, stage_height), fill=(220, 0, 0))
            writer.append_data(np.asarray(img.convert("RGB")))
        writer.close()
        shutil.move(temp_video, out_path)
        if audio_path:
            add_audio_to_video(str(out_path), str(audio_path))
    finally:
        if os.path.exists(temp_video):
            os.remove(temp_video)

    return out_path


def html_to_video(
    html_path: str | Path,
    out_path: str | Path,
    audio_path: str | Path | None = None,
    fps: int = DEFAULT_VIDEO_FPS,
    size: tuple[int, int] = DEFAULT_VIDEO_SIZE,
    duration: float | None = None,
    browser_channel: str = "msedge",
) -> Path:
    """Capture an existing audio_text_viz HTML file to MP4."""
    import imageio

    from mucap.render.utils import add_audio_to_video

    sync_playwright = _require_playwright()
    html_path = Path(html_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = size
    duration = max(float(duration or 1.0), 0.05)
    total_frames = max(1, int(math.ceil(duration * fps)))
    temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

    try:
        with sync_playwright() as p:
            launch_kwargs = {"headless": True}
            try:
                browser = p.chromium.launch(channel=browser_channel, **launch_kwargs)
            except Exception:
                browser = p.chromium.launch(**launch_kwargs)

            page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
            page.goto(_file_url(html_path), wait_until="load")
            page.wait_for_timeout(250)
            page.evaluate(
                """() => {
                    const audio = document.querySelector('audio');
                    if (audio) {
                        audio.pause();
                        audio.currentTime = 0;
                    }
                }"""
            )

            writer = imageio.get_writer(
                temp_video,
                fps=fps,
                codec="libx264",
                quality=8,
                macro_block_size=1,
            )
            try:
                for frame_index in range(total_frames):
                    current = frame_index / float(fps)
                    page.evaluate(
                        """(t) => {
                            const audio = document.querySelector('audio');
                            if (audio) {
                                audio.currentTime = t;
                                audio.dispatchEvent(new Event('timeupdate'));
                                audio.dispatchEvent(new Event('seeked'));
                            }
                        }""",
                        current,
                    )
                    page.wait_for_timeout(1)
                    png = page.locator(".stage").screenshot(type="png")
                    frame = imageio.v3.imread(png)
                    frame = _pad_to_macroblock(np.asarray(frame[:, :, :3]))
                    writer.append_data(frame)
            finally:
                writer.close()
                browser.close()

        shutil.move(temp_video, out_path)
        if audio_path:
            add_audio_to_video(str(out_path), str(audio_path))
    finally:
        if os.path.exists(temp_video):
            os.remove(temp_video)

    return out_path


def visualize_captions_video(
    wav_path_or_name: str,
    captions: object,
    out_path: Path | None = None,
    notes_csv: Path = DEFAULT_NOTES_CSV,
    features_csv: Path | None = DEFAULT_FEATURES_CSV,
    feature_row: dict | None = None,
    audio_path: str | Path | None = None,
    html_path: str | Path | None = None,
    copy_audio: bool = True,
    fps: int = DEFAULT_VIDEO_FPS,
    size: tuple[int, int] = DEFAULT_VIDEO_SIZE,
    mode: str = "direct",
) -> Path:
    """Create MP4 from captions.

    mode="direct" is smoother and reuses audio_text_viz's Python visual
    calculations. mode="html" captures the exact browser-rendered HTML.
    """
    file_name = _audio_key(wav_path_or_name)
    if out_path is None:
        DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
        suffix = "_direct" if mode == "direct" else ""
        out_path = DEFAULT_OUT_DIR / (Path(file_name).stem + f"{suffix}.mp4")
    out_path = Path(out_path)

    if mode == "direct":
        return _direct_video(
            wav_path_or_name,
            captions,
            out_path,
            notes_csv=notes_csv,
            features_csv=features_csv,
            feature_row=feature_row,
            audio_path=audio_path,
            fps=fps,
            size=size,
        )

    if html_path is None:
        html_path = out_path.with_suffix(".html")
        visualize_captions(
            wav_path_or_name,
            captions,
            out_path=Path(html_path),
            notes_csv=notes_csv,
            features_csv=features_csv,
            feature_row=feature_row,
            audio_path=audio_path,
            copy_audio=copy_audio,
        )
    else:
        html_path = Path(html_path)

    parsed = parse_captions(captions)
    notes, _ = load_song(file_name, notes_csv, features_csv, feature_row)
    duration = _duration_from_captions(parsed, note_table(notes))
    return html_to_video(
        html_path,
        out_path,
        audio_path=audio_path,
        fps=fps,
        size=size,
        duration=duration,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notes-csv", type=Path, default=DEFAULT_NOTES_CSV)
    parser.add_argument("--features-csv", type=Path, default=DEFAULT_FEATURES_CSV)
    parser.add_argument("--file-name", required=True)
    parser.add_argument("--captions-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="HTML output path")
    parser.add_argument("--out-video", type=Path, default=None)
    parser.add_argument("--audio-path", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=DEFAULT_VIDEO_FPS)
    parser.add_argument("--mode", choices=("direct", "html"), default="direct")
    args = parser.parse_args()

    html_out = visualize_captions(
        args.file_name,
        args.captions_json,
        out_path=args.out,
        notes_csv=args.notes_csv,
        features_csv=args.features_csv,
        audio_path=args.audio_path,
    )
    video_out = visualize_captions_video(
        args.file_name,
        args.captions_json,
        out_path=(
            args.out_video
            or html_out.with_name(
                html_out.stem + ("_direct" if args.mode == "direct" else "") + ".mp4"
            )
        ),
        notes_csv=args.notes_csv,
        features_csv=args.features_csv,
        audio_path=args.audio_path,
        html_path=html_out,
        fps=args.fps,
        mode=args.mode,
    )
    print(f"html={html_out}")
    print(f"video={video_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
