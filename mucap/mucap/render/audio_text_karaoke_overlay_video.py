"""Overlay an audio-text karaoke MP4 on its matching performance video."""

from __future__ import annotations

import argparse
from pathlib import Path

from mucap.render.audio_text_overlay_video import (
    compose_audio_text_overlay,
    find_source_video,
)


def render_karaoke_overlay(
    wav_path_or_name: str | Path,
    karaoke_video_path: str | Path,
    output_path: str | Path,
) -> Path | None:
    source_video = find_source_video(wav_path_or_name)
    if source_video is None:
        print(
            f"Skipping karaoke overlay: no matching source MP4 for "
            f"{Path(wav_path_or_name).stem}"
        )
        return None
    return compose_audio_text_overlay(
        source_video,
        karaoke_video_path,
        output_path,
        panel_height=40,
        bottom_offset=90,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Overlay a karaoke render on its matching source MP4."
    )
    parser.add_argument("--input_file", required=True, help="Source WAV path")
    parser.add_argument("--text_video", required=True, help="Karaoke MP4 path")
    parser.add_argument("--output", required=True, help="Output overlay MP4 path")
    args = parser.parse_args()
    result = render_karaoke_overlay(
        args.input_file,
        args.text_video,
        args.output,
    )
    if result:
        print(f"Saved karaoke overlay video: {result}")
