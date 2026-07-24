"""Overlay the plain black caption render like the karaoke overlay."""

from __future__ import annotations

import argparse
from pathlib import Path

from mucap.render.audio_text_overlay_video import (
    compose_audio_text_overlay,
    find_source_video,
)


def render_plain_overlay(
    wav_path_or_name: str | Path,
    plain_video_path: str | Path,
    output_path: str | Path,
) -> Path | None:
    source_video = find_source_video(wav_path_or_name)
    if source_video is None:
        print(
            f"Skipping plain overlay: no matching source MP4 for "
            f"{Path(wav_path_or_name).stem}"
        )
        return None
    return compose_audio_text_overlay(
        source_video,
        plain_video_path,
        output_path,
        panel_height=42,
        bottom_offset=90,
        panel_width=1240,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Overlay a plain caption render on its matching source MP4."
    )
    parser.add_argument("--input_file", required=True, help="Source WAV path")
    parser.add_argument("--text_video", required=True, help="Plain MP4 path")
    parser.add_argument("--output", required=True, help="Output overlay MP4 path")
    args = parser.parse_args()
    result = render_plain_overlay(
        args.input_file,
        args.text_video,
        args.output,
    )
    if result:
        print(f"Saved plain overlay video: {result}")
