import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
COLOR_ORDER = ["yellow", "green", "blue", "magenta", "orange", "cyan", "purple", "teal"]


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = ROOT / value
    return value.resolve()


def run_step(name: str, command: list[str], dry_run: bool) -> None:
    print(f"\n=== {name} ===")
    print(" ".join(str(part) for part in command))
    if dry_run:
        return
    subprocess.run(command, cwd=ROOT, check=True)


def load_env_file(path: Path | None) -> None:
    if path is None or not path.exists():
        return

    with path.open("r", encoding="utf-8-sig") as fin:
        for line in fin:
            text = line.strip()
            if not text or text.startswith("#") or "=" not in text:
                continue
            key, value = text.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def slugify_instrument(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug:
        raise ValueError(f"Invalid empty instrument name: {value!r}")
    return slug


def count_entries_from_description(path: Path) -> list[tuple[str, int]]:
    with path.open("r", encoding="utf-8-sig") as fin:
        data = json.load(fin)

    if isinstance(data, dict) and "instruments" in data:
        data = data["instruments"]
    elif isinstance(data, dict) and "objects" in data:
        data = data["objects"]

    if isinstance(data, dict):
        entries = []
        for instrument, count in data.items():
            entries.append((str(instrument), int(count)))
        return entries

    if isinstance(data, list):
        entries = []
        for index, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"Object description entry {index} must be an object")
            instrument = item.get("instrument") or item.get("name") or item.get("class")
            if not instrument:
                raise ValueError(f"Object description entry {index} is missing instrument")
            entries.append((str(instrument), int(item.get("count", 1))))
        return entries

    raise ValueError(
        "Object description JSON must be either {'instruments': {'violin': 3}} "
        "or [{'instrument': 'violin', 'count': 3}]."
    )


def build_object_plan_from_description(description_path: Path) -> dict[str, list[dict[str, str]]]:
    entries = count_entries_from_description(description_path)
    objects: list[dict[str, str]] = []
    color_index = 0

    for instrument_name, count in entries:
        if count <= 0:
            raise ValueError(f"Instrument count must be positive: {instrument_name}={count}")
        instrument = slugify_instrument(instrument_name)
        for number in range(1, count + 1):
            if color_index >= len(COLOR_ORDER):
                raise ValueError(
                    f"Too many objects ({color_index + 1}). This pipeline currently supports "
                    f"{len(COLOR_ORDER)} fixed mask colors: {', '.join(COLOR_ORDER)}."
                )
            object_id = f"{instrument}_{number}"
            objects.append(
                {
                    "object_id": object_id,
                    "instrument": instrument,
                    "description": f"{instrument.replace('_', ' ')} player {number}",
                    "color": COLOR_ORDER[color_index],
                }
            )
            color_index += 1

    if not objects:
        raise ValueError(f"No objects found in object description: {description_path}")
    return {"objects": objects}


def ensure_object_plan(
    clip_dir: Path,
    object_plan: Path | None,
    object_description: Path | None,
    dry_run: bool,
) -> Path:
    target = clip_dir / "detections" / "object_plan.json"
    if object_plan is not None and object_description is not None:
        raise ValueError("Pass either --object-plan or --object-description, not both.")

    if object_description is not None:
        source = object_description.resolve()
        if not source.exists():
            raise FileNotFoundError(f"Object description does not exist: {source}")
        plan = build_object_plan_from_description(source)
        print(f"Create object plan from description: {source} -> {target}")
        print(f"Color order: {', '.join(COLOR_ORDER)}")
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8") as fout:
                json.dump(plan, fout, indent=2, ensure_ascii=False)
                fout.write("\n")
        return target

    if object_plan is None:
        if target.exists():
            return target
        raise FileNotFoundError(
            "Missing object list. Pass --object-description <counts.json> to auto-create "
            "an object plan, or pass --object-plan <path> for a full hand-written plan."
        )

    source = object_plan.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Object plan does not exist: {source}")
    if source == target.resolve():
        return target

    print(f"Copy object plan: {source} -> {target}")
    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the object-audio pipeline from one MP4 clip to per-object "
            "sounding-time coordinate CSV files."
        )
    )
    parser.add_argument("--input", required=True, help="Input .mp4 video path.")
    parser.add_argument(
        "--object-plan",
        type=Path,
        default=None,
        help="Optional object_plan.json to copy into <clip>/detections/object_plan.json.",
    )
    parser.add_argument(
        "--object-description",
        type=Path,
        default=None,
        help=(
            "Optional JSON with instrument counts, e.g. "
            "{\"instruments\": {\"violin\": 3, \"flute\": 1, \"cello\": 1}}. "
            "The wrapper expands it into object_plan.json with fixed colors."
        ),
    )
    parser.add_argument("--assets-root", type=Path, default=Path("outputs/video_assets"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-mask-edit",
        action="store_true",
        help="Use existing detections/colored_on_gray frames instead of calling GPT image edit.",
    )
    parser.add_argument("--mask-edit-model", default="gpt-image-2")
    parser.add_argument("--mask-edit-limit", type=int, default=0)
    parser.add_argument("--mask-edit-sleep", type=float, default=0.0)
    parser.add_argument("--separation-model-id", default="facebook/sam-audio-large-tv")
    parser.add_argument("--reranking-candidates", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--separation-device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--separation-max-side",
        type=int,
        default=0,
        help="Resize frames only for SAM-Audio separation; coordinate CSVs stay in source frame space.",
    )
    parser.add_argument("--max-objects", type=int, default=8)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--local-only", action="store_true", help="Use local RMS only for final matching.")
    parser.add_argument("--realtime-model", default="gpt-realtime")
    parser.add_argument("--skip-realtime-below-threshold", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(resolve_path(args.env_file) if args.env_file else None)
    video = resolve_path(args.input)
    if not video.exists():
        raise FileNotFoundError(f"Input video does not exist: {video}")

    assets_root = resolve_path(args.assets_root)
    clip_id = video.stem
    clip_dir = assets_root / clip_id
    audio_path = clip_dir / "audio" / f"{clip_id}.wav"
    frames_dir = clip_dir / f"frames_{str(args.interval).replace('.', '_')}s"
    gray_frames_dir = clip_dir / "detections" / "grayscale_reference"
    colored_on_gray_dir = clip_dir / "detections" / "colored_on_gray"
    manifest_path = clip_dir / "manifest.json"

    object_plan = ensure_object_plan(
        clip_dir=clip_dir,
        object_plan=resolve_path(args.object_plan) if args.object_plan else None,
        object_description=resolve_path(args.object_description) if args.object_description else None,
        dry_run=args.dry_run,
    )

    extract_audio = [
        sys.executable,
        str(SCRIPT_DIR / "extract_audio_wav.py"),
        "--input",
        str(video),
        "--output",
        str(audio_path),
    ]
    extract_frames = [
        sys.executable,
        str(SCRIPT_DIR / "extract_frames_0_5s.py"),
        "--input",
        str(video),
        "--output-dir",
        str(frames_dir),
        "--interval",
        str(args.interval),
    ]
    if args.overwrite:
        extract_audio.append("--overwrite")
        extract_frames.append("--overwrite")

    run_step("Extract audio", extract_audio, args.dry_run)
    run_step("Extract frames", extract_frames, args.dry_run)

    grayscale_frames = [
        sys.executable,
        str(SCRIPT_DIR / "create_grayscale_frames.py"),
        "--frames-dir",
        str(frames_dir),
        "--output-dir",
        str(gray_frames_dir),
    ]
    if args.overwrite:
        grayscale_frames.append("--overwrite")
    run_step("Create grayscale reference frames", grayscale_frames, args.dry_run)

    if not args.skip_mask_edit:
        edit_masks = [
            sys.executable,
            str(SCRIPT_DIR / "edit_object_masks_gpt.py"),
            "--frames-dir",
            str(frames_dir),
            "--object-plan",
            str(object_plan),
            "--output-dir",
            str(colored_on_gray_dir),
            "--model",
            args.mask_edit_model,
        ]
        if not args.overwrite:
            edit_masks.append("--skip-existing")
        if args.mask_edit_limit > 0:
            edit_masks.extend(["--limit", str(args.mask_edit_limit)])
        if args.mask_edit_sleep > 0:
            edit_masks.extend(["--sleep", str(args.mask_edit_sleep)])
        run_step("Create GPT colored masks", edit_masks, args.dry_run)

    normalize_masks = [
        sys.executable,
        str(SCRIPT_DIR / "normalize_gpt_masks.py"),
        "--frames-dir",
        str(frames_dir),
        "--gray-frames-dir",
        str(gray_frames_dir),
        "--colored-on-gray-dir",
        str(colored_on_gray_dir),
        "--object-plan",
        str(object_plan),
        "--background-mode",
        "black",
        "--write-metadata",
    ]
    run_step("Normalize object masks", normalize_masks, args.dry_run)

    create_manifest = [
        sys.executable,
        str(SCRIPT_DIR / "create_manifest.py"),
        "--clip-dir",
        str(clip_dir),
        "--outputs-root",
        str(assets_root),
        "--overwrite",
    ]
    run_step("Create manifest", create_manifest, args.dry_run)

    separate_audio = [
        sys.executable,
        str(SCRIPT_DIR / "separate_visual_objects.py"),
        "--manifest",
        str(manifest_path),
        "--model-id",
        args.separation_model_id,
        "--reranking-candidates",
        str(args.reranking_candidates),
        "--max-frames",
        str(args.max_frames),
        "--device",
        args.separation_device,
        "--max-side",
        str(args.separation_max_side),
        "--max-objects",
        str(args.max_objects),
    ]
    if args.amp:
        separate_audio.append("--amp")
    run_step("Separate object audio", separate_audio, args.dry_run)

    coords = [
        sys.executable,
        str(SCRIPT_DIR / "frame_mask_coords.py"),
        "--manifest",
        str(manifest_path),
        "--frame-interval",
        str(args.interval),
        "--gray-frames-dir",
        str(gray_frames_dir),
    ]
    run_step("Export object coordinates", coords, args.dry_run)

    match_audio = [
        sys.executable,
        str(SCRIPT_DIR / "match_object_audio_realtime.py"),
        "--manifest",
        str(manifest_path),
        "--model",
        args.realtime_model,
        "--frame-interval",
        str(args.interval),
    ]
    if args.local_only:
        match_audio.append("--local-only")
    if args.skip_realtime_below_threshold:
        match_audio.append("--skip-realtime-below-threshold")
    run_step("Match object audio to coordinates", match_audio, args.dry_run)

    full_csv = clip_dir / "matching" / "time_object_audio_coords_full.csv"
    active_csv = clip_dir / "matching" / "time_object_audio_coords_active_only.csv"
    print("\nDone.")
    print(f"Full CSV: {full_csv}")
    print(f"Active-only CSV: {active_csv}")


if __name__ == "__main__":
    main()
