"""Render plain black caption tokens on karaoke-sized pages."""

from __future__ import annotations

import math
import os
import shutil
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mucap.render.audio_text_bounce_video import _captions_from_result_json
from mucap.render.audio_text_karaoke_video import (
    BASELINE_FROM_BOTTOM_PX,
    KARAOKE_FONT_SIZE,
    KARAOKE_PAGE_SIDE_MARGIN_PX,
    KARAOKE_VIDEO_SIZE,
)
from mucap.render.audio_text_viz import (
    DEFAULT_FEATURES_CSV,
    DEFAULT_NOTES_CSV,
    parse_captions,
)
from mucap.render.audio_text_viz_video import (
    DEFAULT_VIDEO_FPS,
    _font_for_size,
    _text_size,
)

PLAIN_CAPTION_GAP_PX = 28


@dataclass(frozen=True)
class PlainCaption:
    timestamp: float
    text: str


def _plain_duration(
    captions: list[dict],
    audio_path: str | Path | None,
) -> float:
    """Use WAV duration when available; otherwise extend past the final token."""
    if audio_path and Path(audio_path).suffix.lower() == ".wav":
        try:
            with wave.open(str(audio_path), "rb") as wav_file:
                return wav_file.getnframes() / float(wav_file.getframerate())
        except (OSError, EOFError, wave.Error):
            pass
    return max(float(caption["time"]) for caption in captions) + 1.0


def _paginate_events(
    events,
    draw,
    font,
    max_width: float,
    preserve_spaces: bool = True,
):
    pages = []
    page = []
    page_width = 0.0
    space_width = float(PLAIN_CAPTION_GAP_PX) if preserve_spaces else 0.0
    for event in events:
        event_width = float(_text_size(draw, event.text, font)[0])
        separator_width = space_width if page else 0.0
        if page and page_width + separator_width + event_width > max_width:
            pages.append(page)
            page = [event]
            page_width = event_width
        else:
            page.append(event)
            page_width += separator_width + event_width
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
    preserve_spaces: bool = False,
) -> Path:
    """Create a plain black caption MP4, optionally preserving token spaces."""
    import imageio
    from PIL import Image, ImageDraw

    from mucap.render.utils import add_audio_to_video

    parsed_captions = parse_captions(captions)
    if not parsed_captions:
        raise ValueError(f"No captions available for plain video: {wav_path_or_name}")

    duration = _plain_duration(parsed_captions, audio_path)
    events = [
        PlainCaption(timestamp=float(caption["time"]), text=str(caption["text"]))
        for caption in parsed_captions
    ]
    width, height = size
    baseline_y = height - BASELINE_FROM_BOTTOM_PX
    font = _font_for_size(KARAOKE_FONT_SIZE)
    sizing_image = Image.new("RGB", (width, height), "white")
    sizing_draw = ImageDraw.Draw(sizing_image)
    pages = _paginate_events(
        events,
        sizing_draw,
        font,
        max_width=width - (2 * KARAOKE_PAGE_SIDE_MARGIN_PX),
        preserve_spaces=preserve_spaces,
    )
    space_width = float(PLAIN_CAPTION_GAP_PX) if preserve_spaces else 0.0

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
            for event_index, event in enumerate(pages[page_index] if pages else []):
                if event_index:
                    cursor_x += space_width
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
    parser.add_argument(
        "--remove-spaces",
        action="store_true",
        help="Use the legacy rendering that removes spaces between caption tokens.",
    )
    parser.add_argument(
        "--preserve-spaces",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    captions = _captions_from_result_json(args.caption_json)
    output = visualize_captions_plain_video(
        args.input_file,
        captions,
        out_path=args.output_path,
        audio_path=args.input_file,
        fps=args.fps,
        preserve_spaces=not args.remove_spaces,
    )
    print(f"Saved plain audio text video: {output}")


if __name__ == "__main__":
    main()
