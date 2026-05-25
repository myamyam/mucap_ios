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
COLOR_ORDER = [
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
    [255, 128, 0],
    [128, 0, 255],
]


def default_mucap_root() -> Path:
    sibling_mucap = ROOT.parent / "mucap"
    if sibling_mucap.exists():
        return sibling_mucap
    return ROOT.parent / "MuCap-main" / "MuCap-main"


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


def run_step_with_cwd(name: str, command: list[str], cwd: Path, dry_run: bool) -> None:
    print(f"\n=== {name} ===")
    print(f"cwd: {cwd}")
    print(" ".join(str(part) for part in command))
    if dry_run:
        return
    subprocess.run(command, cwd=cwd, check=True)


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


def format_color(color: list[int]) -> str:
    return f"RGB({color[0]}, {color[1]}, {color[2]})"


def build_object_plan_from_description(description_path: Path) -> dict[str, list[dict[str, object]]]:
    entries = count_entries_from_description(description_path)
    objects: list[dict[str, object]] = []
    color_index = 0

    for instrument_name, count in entries:
        if count <= 0:
            raise ValueError(f"Instrument count must be positive: {instrument_name}={count}")
        instrument = slugify_instrument(instrument_name)
        for number in range(1, count + 1):
            if color_index >= len(COLOR_ORDER):
                raise ValueError(
                    f"Too many objects ({color_index + 1}). This pipeline currently supports "
                    f"{len(COLOR_ORDER)} fixed mask colors: "
                    f"{', '.join(format_color(color) for color in COLOR_ORDER)}."
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
        print(f"Color order: {', '.join(format_color(color) for color in COLOR_ORDER)}")
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
            "The wrapper expands it into object_plan.json with fixed RGB colors."
        ),
    )
    parser.add_argument("--assets-root", type=Path, default=Path("outputs/video_assets"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-mask-edit",
        action="store_true",
        help="Use existing detections/colored_on_black frames instead of calling GPT image edit.",
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
    parser.add_argument(
        "--prepare-mucap-inputs",
        action="store_true",
        help="Copy separated object wav files into <clip>/captions/mucap_input/<clip_id> after the normal pipeline.",
    )
    parser.add_argument(
        "--mucap-input-dir",
        type=Path,
        default=None,
        help="Override MuCap clip input directory. Default: <clip>/captions/mucap_input/<clip_id>.",
    )
    parser.add_argument(
        "--run-mucap-captions",
        action="store_true",
        help="Run MuCap caption-only JSON/TXT generation after preparing MuCap inputs.",
    )
    parser.add_argument(
        "--mucap-root",
        type=Path,
        default=default_mucap_root(),
        help=(
            "MuCap repository root containing mucap/generate_caption_batch.py. "
            "Default prefers ../mucap, then falls back to ../MuCap-main/MuCap-main."
        ),
    )
    parser.add_argument(
        "--mucap-output-dir",
        type=Path,
        default=None,
        help="Override MuCap output directory. Default: <clip>/captions/mucap_output.",
    )
    parser.add_argument("--mucap-num-workers", type=int, default=1)
    parser.add_argument("--mucap-model", default="gpt-audio")
    parser.add_argument("--mucap-use-audio-feature", action="store_true")
    parser.add_argument("--mucap-use-rag", action="store_true")
    parser.add_argument(
        "--render-captioned-video",
        action="store_true",
        help="Render final captioned MP4 after MuCap caption JSON generation.",
    )
    parser.add_argument(
        "--captioned-output",
        type=Path,
        default=None,
        help="Override final captioned MP4 path. Default: <clip>/final/captioned_video.mp4.",
    )
    parser.add_argument(
        "--caption-position-mode",
        choices=["dynamic", "fixed-by-scene"],
        default="dynamic",
        help="Final caption position mode. fixed-by-scene uses median bbox per object per detected scene.",
    )
    parser.add_argument(
        "--scene-cut-aware-render",
        action="store_true",
        help="Detect scene cuts and disable final caption bbox interpolation across cuts.",
    )
    parser.add_argument("--scene-cut-threshold", type=float, default=0.5)
    parser.add_argument("--scene-cut-min-gap", type=float, default=0.4)
    parser.add_argument("--scene-cut-frame-step", type=int, default=1)
    parser.add_argument(
        "--scene-cuts-output",
        type=Path,
        default=None,
        help="Override scene cut JSON path. Default: <clip>/scene_cuts.json.",
    )
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
    colored_frames_dir = clip_dir / "detections" / "colored_on_black"
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

    if not args.skip_mask_edit:
        edit_masks = [
            sys.executable,
            str(SCRIPT_DIR / "edit_object_masks_gpt.py"),
            "--frames-dir",
            str(frames_dir),
            "--object-plan",
            str(object_plan),
            "--output-dir",
            str(colored_frames_dir),
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
        "--black-tolerance",
        "30",
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

    if args.run_mucap_captions or args.render_captioned_video:
        args.prepare_mucap_inputs = True

    mucap_input_dir = None
    mucap_batch_input_dir = None
    if args.prepare_mucap_inputs:
        prepare_mucap = [
            sys.executable,
            str(SCRIPT_DIR / "prepare_mucap_inputs.py"),
            "--manifest",
            str(manifest_path),
        ]
        if args.mucap_input_dir is not None:
            prepare_mucap.extend(["--output-dir", str(resolve_path(args.mucap_input_dir))])
        if args.overwrite:
            prepare_mucap.append("--overwrite")
        run_step("Prepare MuCap inputs", prepare_mucap, args.dry_run)
        mucap_input_dir = (
            resolve_path(args.mucap_input_dir)
            if args.mucap_input_dir is not None
            else clip_dir / "captions" / "mucap_input" / clip_id
        )
        mucap_batch_input_dir = mucap_input_dir.parent

    mucap_output_dir = (
        resolve_path(args.mucap_output_dir)
        if args.mucap_output_dir is not None
        else clip_dir / "captions" / "mucap_output"
    )

    if args.run_mucap_captions:
        mucap_root = resolve_path(args.mucap_root)
        if not args.dry_run and not (mucap_root / "mucap" / "generate_caption_batch.py").exists():
            raise FileNotFoundError(f"MuCap root does not look valid: {mucap_root}")
        if mucap_batch_input_dir is None:
            raise RuntimeError("Internal error: MuCap input directory was not prepared.")
        mucap_caption = [
            sys.executable,
            "-m",
            "mucap.generate_caption_batch",
            "--input_dir",
            str(mucap_batch_input_dir),
            "--output_dir",
            str(mucap_output_dir),
            "--caption_only",
            "true",
            "--use_audio_feature",
            "true" if args.mucap_use_audio_feature else "false",
            "--use_rag",
            "true" if args.mucap_use_rag else "false",
            "--num_workers",
            str(args.mucap_num_workers),
            "--model",
            args.mucap_model,
        ]
        run_step_with_cwd("Generate MuCap caption JSON", mucap_caption, mucap_root, args.dry_run)

    captioned_output = (
        resolve_path(args.captioned_output)
        if args.captioned_output is not None
        else clip_dir / "final" / "captioned_video.mp4"
    )
    scene_cuts_output = (
        resolve_path(args.scene_cuts_output)
        if args.scene_cuts_output is not None
        else clip_dir / "scene_cuts.json"
    )

    if args.render_captioned_video:
        needs_scene_cuts = args.scene_cut_aware_render or args.caption_position_mode == "fixed-by-scene"
        if needs_scene_cuts:
            detect_scene_cuts = [
                sys.executable,
                str(SCRIPT_DIR / "detect_scene_cuts.py"),
                "--input",
                str(video),
                "--output",
                str(scene_cuts_output),
                "--threshold",
                str(args.scene_cut_threshold),
                "--min-gap",
                str(args.scene_cut_min_gap),
                "--frame-step",
                str(args.scene_cut_frame_step),
            ]
            run_step("Detect scene cuts", detect_scene_cuts, args.dry_run)

        render_captioned = [
            sys.executable,
            str(SCRIPT_DIR / "render_mucap_object_captions.py"),
            "--clip-id",
            clip_id,
            "--input-video",
            str(video),
            "--matching-csv",
            str(clip_dir / "matching" / "time_object_audio_coords_full.csv"),
            "--mucap-output",
            str(mucap_output_dir),
            "--output",
            str(captioned_output),
            "--position-mode",
            args.caption_position_mode,
        ]
        if args.scene_cut_aware_render:
            render_captioned.extend(["--scene-cut-aware", "--scene-cuts", str(scene_cuts_output)])
        elif args.caption_position_mode == "fixed-by-scene":
            render_captioned.extend(["--scene-cuts", str(scene_cuts_output)])
        run_step("Render captioned video", render_captioned, args.dry_run)

    full_csv = clip_dir / "matching" / "time_object_audio_coords_full.csv"
    active_csv = clip_dir / "matching" / "time_object_audio_coords_active_only.csv"
    print("\nDone.")
    print(f"Full CSV: {full_csv}")
    print(f"Active-only CSV: {active_csv}")
    if mucap_input_dir is not None:
        print(f"MuCap input clip dir: {mucap_input_dir}")
        print(f"MuCap batch input dir: {mucap_input_dir.parent}")
    if args.run_mucap_captions:
        print(f"MuCap output dir: {mucap_output_dir}")
    if args.render_captioned_video:
        print(f"Captioned MP4: {captioned_output}")
        if args.scene_cut_aware_render or args.caption_position_mode == "fixed-by-scene":
            print(f"Scene cuts: {scene_cuts_output}")


if __name__ == "__main__":
    main()
