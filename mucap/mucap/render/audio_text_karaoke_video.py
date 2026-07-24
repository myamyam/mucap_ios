"""Render caption pages as a karaoke-style color fill on one baseline.

Each page is filled by actual text width and switches when the first caption on
the next page reaches its timestamp. Before the internal playhead reaches a
glyph it is drawn with very low opacity; the playhead reveals its established
audio_text_viz color from left to right.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from mucap.render.audio_text_bounce_video import (
    CAPTION_GAP_PX,
    _captions_from_result_json,
    _flow_layout,
    _paginate_events,
    _prepare_events,
    _runtime_feature_row_if_needed,
)
from mucap.render.audio_text_viz import (
    CAPTION_MAPPING,
    DEFAULT_FEATURES_CSV,
    DEFAULT_NOTES_CSV,
    _audio_key,
    _duration_from_captions,
    _opacity_style,
    _pitch_norm,
    _visual_mode_from_context,
    apply_note_mapping,
    load_song,
    note_table,
    parse_captions,
)
from mucap.render.audio_text_viz_video import (
    DEFAULT_VIDEO_FPS,
    _font_for_size,
    _rgb_for_row,
    _text_size,
)


PENDING_OPACITY_FACTOR = 0.10
KARAOKE_VIDEO_SIZE = (1240, 42)
BASELINE_FROM_BOTTOM_PX = 10
KARAOKE_PAGE_SIDE_MARGIN_PX = 20
KARAOKE_FONT_SIZE = 30


def _draw_karaoke_group(
    img,
    text: str,
    x: float,
    baseline_y: float,
    font,
    rgb: tuple[int, int, int],
    final_opacity: float,
    playhead_x: float,
) -> None:
    """Draw a dim full glyph layer and reveal the final layer left-to-right."""
    from PIL import Image, ImageDraw

    pending_alpha = int(
        np.clip(final_opacity * PENDING_OPACITY_FACTOR, 0.0, 1.0) * 255
    )
    final_alpha = int(np.clip(final_opacity, 0.0, 1.0) * 255)

    pending = Image.new("RGBA", img.size, (255, 255, 255, 0))
    pending_draw = ImageDraw.Draw(pending)
    pending_draw.text(
        (x, baseline_y),
        text,
        font=font,
        fill=(*rgb, pending_alpha),
        anchor="ls",
    )
    img.alpha_composite(pending)

    clip_right = int(round(np.clip(playhead_x, 0.0, float(img.width))))
    if clip_right <= 0:
        return

    final = Image.new("RGBA", img.size, (255, 255, 255, 0))
    final_draw = ImageDraw.Draw(final)
    final_draw.text(
        (x, baseline_y),
        text,
        font=font,
        fill=(*rgb, final_alpha),
        anchor="ls",
    )
    revealed = final.crop((0, 0, clip_right, img.height))
    img.alpha_composite(revealed, dest=(0, 0))


def visualize_captions_karaoke_video(
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
    """Create an additional MP4 with playhead-driven caption color filling."""
    import imageio
    from PIL import Image, ImageDraw

    from mucap.render.utils import add_audio_to_video

    file_name = _audio_key(wav_path_or_name)
    notes, song_row = load_song(file_name, notes_csv, features_csv, feature_row)
    df = apply_note_mapping(note_table(notes), CAPTION_MAPPING)
    parsed_captions = parse_captions(captions)
    if df.empty:
        raise ValueError(f"No note rows available for karaoke video: {file_name}")
    if not parsed_captions:
        raise ValueError(f"No captions available for karaoke video: {file_name}")

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
    stage_height = height - 34
    baseline_y = height - BASELINE_FROM_BOTTOM_PX
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    font = _font_for_size(KARAOKE_FONT_SIZE)
    sizing_image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    sizing_draw = ImageDraw.Draw(sizing_image)
    pages = _paginate_events(
        events,
        sizing_draw,
        font,
        max_width=width - (2 * KARAOKE_PAGE_SIDE_MARGIN_PX),
        gap_px=CAPTION_GAP_PX,
    )

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

            img = Image.new("RGBA", (width, height), (255, 255, 255, 255))
            draw = ImageDraw.Draw(img)

            page_index = 0
            for index, page in enumerate(pages[1:], start=1):
                if current >= page[0].timestamp:
                    page_index = index
                else:
                    break
            visible_events = pages[page_index] if pages else []
            event_centers, _ = _flow_layout(
                visible_events,
                draw,
                font,
                gap_px=CAPTION_GAP_PX,
            )
            event_times = np.asarray(
                [event.timestamp for event in visible_events],
                dtype=float,
            )
            playhead_times = event_times
            playhead_centers = np.asarray(event_centers, dtype=float)
            if visible_events:
                last_event_width = _text_size(
                    draw,
                    visible_events[-1].text,
                    font,
                )[0]
                last_right_edge = (
                    event_centers[-1] + last_event_width / 2.0
                )
                if page_index + 1 < len(pages):
                    page_fill_end = pages[page_index + 1][0].timestamp
                else:
                    page_fill_end = duration
                page_fill_end = max(
                    float(page_fill_end) - (1.0 / float(fps)),
                    float(event_times[-1]),
                )
                playhead_times = np.append(event_times, page_fill_end)
                playhead_centers = np.append(
                    playhead_centers,
                    last_right_edge,
                )
            playhead_x = float(np.interp(
                current,
                playhead_times,
                playhead_centers,
            )) + KARAOKE_PAGE_SIDE_MARGIN_PX

            for event, event_center in zip(visible_events, event_centers):
                x_center = event_center + KARAOKE_PAGE_SIDE_MARGIN_PX
                event_width = _text_size(draw, event.text, font)[0]
                if x_center + event_width / 2.0 < 0.0:
                    continue
                if x_center - event_width / 2.0 > width:
                    continue
                group_widths = [
                    _text_size(draw, group.text, font)[0]
                    for group in event.groups
                ]
                cursor_x = x_center - sum(group_widths) / 2.0
                base_norm = _pitch_norm(event.base_row, df)

                for group, group_width in zip(event.groups, group_widths):
                    rgb = _rgb_for_row(group.row, df, visual_mode, base_norm)
                    opacity = _opacity_style(group.row, df, base_norm)
                    _draw_karaoke_group(
                        img,
                        group.text,
                        cursor_x,
                        baseline_y,
                        font,
                        rgb,
                        opacity,
                        playhead_x,
                    )
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


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Re-render only the additional audio-text karaoke MP4."
    )
    parser.add_argument("--input_file", required=True, help="Original input WAV")
    parser.add_argument(
        "--caption_json",
        required=True,
        help="Existing MuCap result JSON containing output.caption",
    )
    parser.add_argument("--output_path", required=True, help="Karaoke MP4 output path")
    parser.add_argument("--fps", type=int, default=DEFAULT_VIDEO_FPS)
    args = parser.parse_args()

    captions = _captions_from_result_json(args.caption_json)
    feature_row = _runtime_feature_row_if_needed(args.input_file)
    output = visualize_captions_karaoke_video(
        args.input_file,
        captions,
        out_path=args.output_path,
        feature_row=feature_row,
        audio_path=args.input_file,
        fps=args.fps,
    )
    print(f"Saved audio text karaoke video: {output}")


if __name__ == "__main__":
    main()
