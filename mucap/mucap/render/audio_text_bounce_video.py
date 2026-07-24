"""Render caption text as note-driven jumps from a shared baseline.

This is an additional renderer. It reuses audio_text_viz's note assignment,
palette, font, and timeline calculations without changing the existing HTML or
direct MP4 outputs.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mucap.render.audio_text_viz import (
    CAPTION_MAPPING,
    DEFAULT_FEATURES_CSV,
    DEFAULT_NOTES_CSV,
    _audio_key,
    _caption_end_time,
    _duration_from_captions,
    _opacity_style,
    _pitch_norm,
    _visual_mode_from_context,
    apply_note_mapping,
    load_song,
    note_index_at,
    note_table,
    parse_captions,
)
from mucap.render.audio_text_viz_video import (
    DEFAULT_VIDEO_FPS,
    _font_for_size,
    _rgb_for_row,
    _text_size,
)


MIN_JUMP_PX = 25.0
MAX_JUMP_PX = 75.0
MIN_AIR_TIME = 0.45
MAX_AIR_TIME = 2.50
BOUNCE_VIDEO_SIZE = (1240, 120)
BASELINE_FROM_BOTTOM_PX = 10
UNIFORM_FONT_SIZE = 24
CAPTION_GAP_PX = 24
PAGE_SIDE_MARGIN_PX = 32
BOUNCE_PAGE_SIDE_MARGIN_PX = 20
BOUNCE_UNIFORM_FONT_SIZE = 30
BOUNCE_CAPTION_GAP_PX = 28


@dataclass
class BounceGroup:
    text: str
    note_index: int
    row: pd.Series
    jump_height: float
    air_time: float


@dataclass
class BounceCaption:
    timestamp: float
    text: str
    base_row: pd.Series
    groups: list[BounceGroup]


def _jump_height(row: pd.Series, df: pd.DataFrame) -> float:
    """Map the shared audio_text_viz pitch normalization to jump pixels."""
    norm = float(np.clip(_pitch_norm(row, df), 0.0, 1.0))
    return float(np.interp(norm, [0.0, 1.0], [MIN_JUMP_PX, MAX_JUMP_PX]))


def _air_time(row: pd.Series) -> float:
    """Use the assigned note duration, with visual safety bounds."""
    try:
        duration = float(row.get("duration", MIN_AIR_TIME))
    except (TypeError, ValueError):
        duration = MIN_AIR_TIME
    if not np.isfinite(duration):
        duration = MIN_AIR_TIME
    return float(np.clip(duration, MIN_AIR_TIME, MAX_AIR_TIME))


def _caption_groups(
    text: str,
    timestamp: float,
    end_time: float,
    df: pd.DataFrame,
) -> list[BounceGroup]:
    """Assign characters to notes and merge adjacent characters on one note."""
    chars = list(text)
    if not chars or df.empty:
        return []

    assigned: list[tuple[str, int, pd.Series]] = []
    span = max(end_time - timestamp, 0.05)
    for index, char in enumerate(chars):
        char_time = timestamp + ((index + 0.5) / len(chars)) * span
        note_index = note_index_at(char_time, df)
        assigned.append((char, note_index, df.iloc[note_index]))

    merged: list[tuple[str, int, pd.Series]] = []
    for char, note_index, row in assigned:
        if merged and merged[-1][1] == note_index:
            previous_text, _, previous_row = merged[-1]
            merged[-1] = (previous_text + char, note_index, previous_row)
        else:
            merged.append((char, note_index, row))

    return [
        BounceGroup(
            text=group_text,
            note_index=note_index,
            row=row,
            jump_height=_jump_height(row, df),
            air_time=_air_time(row),
        )
        for group_text, note_index, row in merged
    ]


def _prepare_events(
    captions: list[dict],
    df: pd.DataFrame,
    duration: float,
) -> list[BounceCaption]:
    events: list[BounceCaption] = []
    for index, caption in enumerate(captions):
        timestamp = float(caption["time"])
        text = str(caption["text"])
        base_row = df.iloc[note_index_at(timestamp, df)]
        end_time = _caption_end_time(index, captions, duration)
        events.append(
            BounceCaption(
                timestamp=timestamp,
                text=text,
                base_row=base_row,
                groups=_caption_groups(text, timestamp, end_time, df),
            )
        )
    return events


def _flow_layout(
    events: list[BounceCaption],
    draw,
    font,
    max_width: float | None = None,
    gap_px: float = CAPTION_GAP_PX,
) -> tuple[list[float], float]:
    """Place captions in one line, tightening gaps only when width is limited."""
    caption_widths = [_text_size(draw, event.text, font)[0] for event in events]
    gap = float(gap_px)
    if max_width is not None and len(caption_widths) > 1:
        available_for_gaps = max_width - sum(caption_widths)
        gap = max(0.0, min(gap, available_for_gaps / (len(caption_widths) - 1)))

    centers: list[float] = []
    cursor = 0.0
    for caption_width in caption_widths:
        centers.append(cursor + caption_width / 2.0)
        cursor += caption_width + gap
    total_width = max(0.0, cursor - gap)
    return centers, total_width


def _camera_left(
    current: float,
    events: list[BounceCaption],
    centers: list[float],
    total_width: float,
    viewport_width: float,
) -> float:
    """Scroll the fixed caption line smoothly according to caption timestamps."""
    if not events or not centers or total_width <= viewport_width:
        return 0.0
    times = np.asarray([event.timestamp for event in events], dtype=float)
    positions = np.asarray(centers, dtype=float)
    focus_x = float(np.interp(current, times, positions))
    return float(np.clip(
        focus_x - viewport_width / 2.0,
        0.0,
        total_width - viewport_width,
    ))


def _paginate_events(
    events: list[BounceCaption],
    draw,
    font,
    max_width: float,
    gap_px: float = CAPTION_GAP_PX,
) -> list[list[BounceCaption]]:
    """Fill each page by text width and break before crossing the right margin."""
    pages: list[list[BounceCaption]] = []
    page: list[BounceCaption] = []
    page_width = 0.0

    for event in events:
        event_width = float(_text_size(draw, event.text, font)[0])
        next_width = (
            event_width
            if not page
            else page_width + gap_px + event_width
        )
        would_overflow = bool(page) and next_width > max_width

        if would_overflow:
            pages.append(page)
            page = [event]
            page_width = event_width
        else:
            page.append(event)
            page_width = next_width

    if page:
        pages.append(page)
    return pages


def _page_events(
    pages: list[list[BounceCaption]],
    current: float,
) -> list[BounceCaption]:
    """Switch pages when the first caption on the next page reaches its timestamp."""
    if not pages:
        return []
    page_index = 0
    for index, page in enumerate(pages[1:], start=1):
        if current >= page[0].timestamp:
            page_index = index
        else:
            break
    return pages[page_index]


def _group_y(baseline_y: float, elapsed: float, group: BounceGroup) -> float:
    """Keep text on the baseline except while its timestamp-driven jump is active."""
    if elapsed < 0.0:
        return baseline_y
    if elapsed <= group.air_time:
        progress = elapsed / group.air_time
        return baseline_y - group.jump_height * math.sin(math.pi * progress)
    return baseline_y


def visualize_captions_bounce_video(
    wav_path_or_name: str,
    captions: object,
    out_path: str | Path,
    notes_csv: Path = DEFAULT_NOTES_CSV,
    features_csv: Path | None = DEFAULT_FEATURES_CSV,
    feature_row: dict | None = None,
    audio_path: str | Path | None = None,
    fps: int = DEFAULT_VIDEO_FPS,
    size: tuple[int, int] = BOUNCE_VIDEO_SIZE,
) -> Path:
    """Create an additional MP4 in which caption groups jump by pitch/duration."""
    import imageio
    from PIL import Image, ImageDraw

    from mucap.render.utils import add_audio_to_video

    file_name = _audio_key(wav_path_or_name)
    notes, song_row = load_song(file_name, notes_csv, features_csv, feature_row)
    df = apply_note_mapping(note_table(notes), CAPTION_MAPPING)
    parsed_captions = parse_captions(captions)
    if df.empty:
        raise ValueError(f"No note rows available for bounce video: {file_name}")
    if not parsed_captions:
        raise ValueError(f"No captions available for bounce video: {file_name}")

    context_row = dict(song_row)
    if feature_row:
        context_row.update(feature_row)
    visual_mode = _visual_mode_from_context(
        audio_path or wav_path_or_name,
        context_row,
    )
    duration = _duration_from_captions(parsed_captions, df)
    events = _prepare_events(parsed_captions, df, duration)

    width, height = size
    baseline_y = height - BASELINE_FROM_BOTTOM_PX
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    font = _font_for_size(BOUNCE_UNIFORM_FONT_SIZE)
    sizing_image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    sizing_draw = ImageDraw.Draw(sizing_image)
    pages = _paginate_events(
        events,
        sizing_draw,
        font,
        max_width=width - (2 * BOUNCE_PAGE_SIDE_MARGIN_PX),
        gap_px=BOUNCE_CAPTION_GAP_PX,
    )

    try:
        writer = imageio.get_writer(
            temp_video,
            fps=fps,
            codec="libx264",
            quality=9,
            macro_block_size=8,
        )
        total_frames = max(1, int(math.ceil(duration * fps)))
        for frame_index in range(total_frames):
            current = frame_index / float(fps)

            img = Image.new("RGBA", (width, height), (255, 255, 255, 255))
            draw = ImageDraw.Draw(img)

            visible_events = _page_events(pages, current)
            event_centers, total_text_width = _flow_layout(
                visible_events,
                draw,
                font,
                gap_px=BOUNCE_CAPTION_GAP_PX,
            )
            page_offset_x = float(BOUNCE_PAGE_SIDE_MARGIN_PX)

            for event, event_center in zip(visible_events, event_centers):
                x_center = event_center + page_offset_x
                event_width = _text_size(draw, event.text, font)[0]
                if x_center + event_width / 2.0 < 0.0:
                    continue
                if x_center - event_width / 2.0 > width:
                    continue
                elapsed = current - event.timestamp

                group_widths = [
                    _text_size(draw, group.text, font)[0]
                    for group in event.groups
                ]
                cursor_x = x_center - sum(group_widths) / 2.0
                base_norm = _pitch_norm(event.base_row, df)

                for group, group_width in zip(event.groups, group_widths):
                    y = _group_y(baseline_y, elapsed, group)
                    opacity = _opacity_style(group.row, df, base_norm)
                    rgb = _rgb_for_row(group.row, df, visual_mode, base_norm)
                    alpha = int(np.clip(opacity, 0.0, 1.0) * 255)
                    layer = Image.new("RGBA", img.size, (255, 255, 255, 0))
                    layer_draw = ImageDraw.Draw(layer)
                    layer_draw.text(
                        (cursor_x, y),
                        group.text,
                        font=font,
                        fill=(*rgb, alpha),
                        anchor="ls",
                    )
                    img.alpha_composite(layer)
                    cursor_x += group_width

            writer.append_data(np.asarray(img.convert("RGB")))

        writer.close()
        shutil.move(temp_video, out_path)
        if audio_path:
            add_audio_to_video(str(out_path), str(audio_path))
    finally:
        if os.path.exists(temp_video):
            os.remove(temp_video)

    return out_path


def _captions_from_result_json(path: str | Path) -> list[tuple[float, str]]:
    """Read MuCap's saved LLM caption string and convert it to renderer tokens."""
    from mucap.generate_caption import parse_caption

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    caption = data.get("output", {}).get("caption")
    if not caption:
        raise ValueError(f"No output.caption found in: {path}")
    return parse_caption(str(caption))


def _runtime_feature_row_if_needed(wav_path: str | Path) -> dict | None:
    """Extract notes only when the shared CSV lookup cannot serve this WAV."""
    try:
        load_song(_audio_key(wav_path))
        return None
    except (KeyError, FileNotFoundError):
        from music_features.feature_extractor import AudioFeatureExtraction

        extractor = AudioFeatureExtraction()
        feature_row = extractor.extract_features(str(wav_path))
        try:
            feature_row["top5_instruments"] = extractor.predict_instrument_essentia(
                str(wav_path)
            )
        except Exception as exc:
            print(f"Warning: instrument prediction failed for bounce render: {exc}")
            feature_row["top5_instruments"] = []
        return feature_row


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Re-render only the additional audio-text bounce MP4."
    )
    parser.add_argument("--input_file", required=True, help="Original input WAV")
    parser.add_argument(
        "--caption_json",
        required=True,
        help="Existing MuCap result JSON containing output.caption",
    )
    parser.add_argument("--output_path", required=True, help="Bounce MP4 output path")
    parser.add_argument("--fps", type=int, default=DEFAULT_VIDEO_FPS)
    args = parser.parse_args()

    captions = _captions_from_result_json(args.caption_json)
    feature_row = _runtime_feature_row_if_needed(args.input_file)
    output = visualize_captions_bounce_video(
        args.input_file,
        captions,
        out_path=args.output_path,
        feature_row=feature_row,
        audio_path=args.input_file,
        fps=args.fps,
    )
    print(f"Saved audio text bounce video: {output}")


if __name__ == "__main__":
    main()
