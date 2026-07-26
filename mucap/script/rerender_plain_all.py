"""Re-render only spaced plain captions and their overlays in one output tree."""

from __future__ import annotations

import argparse
from pathlib import Path

from mucap.render.audio_text_bounce_video import _captions_from_result_json
from mucap.render.audio_text_plain_overlay_video import render_plain_overlay
from mucap.render.audio_text_plain_video import visualize_captions_plain_video


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _find_wav(json_path: Path) -> Path:
    """Prefer the WAV copied beside each result, then search known input trees."""
    result_dir = json_path.parent
    local_wavs = sorted(result_dir.glob("*.wav"))
    non_haptic = [path for path in local_wavs if not path.stem.endswith("-haptic")]
    if len(non_haptic) == 1:
        return non_haptic[0]

    result_stem = json_path.stem
    audio_stem = result_stem.removesuffix("-dp_af_rag")
    exact_local = result_dir / f"{audio_stem}.wav"
    if exact_local.is_file():
        return exact_local

    input_roots = (
        PROJECT_ROOT / "input" / "input_wav",
        PROJECT_ROOT / "input" / "mov_wav",
        PROJECT_ROOT / "input",
    )
    for root in input_roots:
        if not root.is_dir():
            continue
        matches = sorted(root.rglob(f"{audio_stem}.wav"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No WAV found for {json_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Re-render only *-audio_text_plain.mp4 and "
            "*-audio_text_plain_overlay.mp4 below an output directory."
        )
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory to scan recursively (for example output_0724).",
    )
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    json_paths = sorted(output_dir.rglob("*-dp_af_rag.json"))
    if not json_paths:
        print(f"No *-dp_af_rag.json results found below {output_dir}")
        return 1

    failures: list[str] = []
    rendered = 0
    for json_path in json_paths:
        prefix = json_path.with_suffix("")
        plain_path = Path(f"{prefix}-audio_text_plain.mp4")
        overlay_path = Path(f"{prefix}-audio_text_plain_overlay.mp4")
        try:
            wav_path = _find_wav(json_path)
            captions = _captions_from_result_json(json_path)
            visualize_captions_plain_video(
                str(wav_path),
                captions,
                out_path=plain_path,
                audio_path=wav_path,
                fps=args.fps,
                preserve_spaces=True,
            )
            overlay_result = render_plain_overlay(
                wav_path,
                plain_path,
                overlay_path,
            )
            rendered += 1
            overlay_status = "overlay rendered" if overlay_result else "overlay skipped"
            print(f"DONE {json_path.parent.name}: plain + {overlay_status}", flush=True)
        except Exception as exc:
            failures.append(f"{json_path}: {exc}")
            print(f"FAILED {json_path}: {exc}", flush=True)

    print(
        f"Completed {rendered}/{len(json_paths)} result(s); "
        f"{len(failures)} failure(s).",
        flush=True,
    )
    if failures:
        for failure in failures:
            print(f"  {failure}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
