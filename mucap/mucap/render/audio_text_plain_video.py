"""Render caption tokens as contiguous black text on karaoke-sized pages."""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from mucap.render.audio_text_bounce_video import (
    _captions_from_result_json,
    _prepare_events,
    _runtime_feature_row_if_needed,
)
from mucap.render.audio_text_karaoke_video import (
    BASELINE_FROM_BOTTOM_PX,
    KARAOKE_FONT_SIZE,
    KARAOKE_PAGE_SIDE_MARGIN_PX,
    KARAOKE_VIDEO_SIZE,
)
from mucap.render.audio_text_viz import (
    CAPTION_MAPPING,
    DEFAULT_FEATURES_CSV,
    DEFAULT_NOTES_CSV,
    _audio_key,
    _duration_from_captions,
    apply_note_mapping,
    load_song,
    note_table,
    parse_captions,
)
from mucap.render.audio_text_viz_video import (
    DEFAULT_VIDEO_FPS,
    _font_for_size,
    _text_size,
)

def _paginate_without_gaps(events, draw, font, max_width: float):
    pages = []
    page = []
    page_width = 0.0
    for event in events:
        event_width = float(_text_size(draw, event.text, font)[0])
        if page and page_width + event_width > max_width:
            pages.append(page)
            page = [event]
            page_width = event_width
        else:
            page.append(event)
            page_width += event_width
    if page:
        pages.append(page)
    return pages


def visualize_captions_plain_video(
    wav_path_or_name: str,
    captions: object,
    out_path: str | Path,
    notes_csv: Path = DEFAULT_NOTES_CSV,
    features_csv: Path | None = DEFAULT_FEATURES_CSV,
    feature_row: dict | None = None,
    audio_path: str | Path | None = None,
    fps: int = DEFAULT_VIDEO_FPS,
    size: tuple[int, int] = KARAOKE_VIDEO_SIZE,
) -> Path:
    """Create a black, whitespace-free caption MP4 beside existing renders."""
    import imageio
    from PIL import Image, ImageDraw

    from mucap.render.utils import add_audio_to_video

    file_name = _audio_key(wav_path_or_name)
    notes, _ = load_song(file_name, notes_csv, features_csv, feature_row)
    df = apply_note_mapping(note_table(notes), CAPTION_MAPPING)
    parsed_captions = parse_captions(captions)
    if df.empty:
        raise ValueError(f"No note rows available for plain video: {file_name}")
    if not parsed_captions:
        raise ValueError(f"No captions available for plain video: {file_name}")

    duration = _duration_from_captions(parsed_captions, df)
    events = _prepare_events(parsed_captions, df, duration)
    width, height = size
    baseline_y = height - BASELINE_FROM_BOTTOM_PX
    font = _font_for_size(KARAOKE_FONT_SIZE)
    sizing_image = Image.new("RGB", (width, height), "white")
    sizing_draw = ImageDraw.Draw(sizing_image)
    pages = _paginate_without_gaps(
        events,
        sizing_draw,
        font,
        max_width=width - (2 * KARAOKE_PAGE_SIDE_MARGIN_PX),
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    try:
        writer = imageio.get_writer(
            temp_video,
            fps=fps,
            codec="libx264",
            quality=9,
            macro_block_size=2,
        )
        total_frames = max(1, int(math.ceil(duration * fps)))
        for frame_index in range(total_frames):
            current = frame_index / float(fps)
            page_index = 0
            for index, page in enumerate(pages[1:], start=1):
                if current >= page[0].timestamp:
                    page_index = index
                else:
                    break

            img = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(img)
            cursor_x = float(KARAOKE_PAGE_SIDE_MARGIN_PX)
            for event in pages[page_index] if pages else []:
                draw.text(
                    (cursor_x, baseline_y),
                    event.text,
                    font=font,
                    fill=(0, 0, 0),
                    anchor="ls",
                )
                cursor_x += _text_size(draw, event.text, font)[0]
            writer.append_data(np.asarray(img))

        writer.close()
        shutil.move(temp_video, out_path)
        if audio_path:
            add_audio_to_video(str(out_path), str(audio_path))
    finally:
        if os.path.exists(temp_video):
            os.remove(temp_video)

    return out_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Render whitespace-free black captions from a MuCap result."
    )
    parser.add_argument("--input_file", required=True, help="Original input WAV")
    parser.add_argument("--caption_json", required=True, help="MuCap result JSON")
    parser.add_argument("--output_path", required=True, help="Plain MP4 output path")
    parser.add_argument("--fps", type=int, default=DEFAULT_VIDEO_FPS)
    args = parser.parse_args()

    captions = _captions_from_result_json(args.caption_json)
    feature_row = _runtime_feature_row_if_needed(args.input_file)
    output = visualize_captions_plain_video(
        args.input_file,
        captions,
        out_path=args.output_path,
        feature_row=feature_row,
        audio_path=args.input_file,
        fps=args.fps,
    )
    print(f"Saved plain audio text video: {output}")


if __name__ == "__main__":
    main()
