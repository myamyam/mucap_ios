import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fin:
        data = json.load(fin)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def resolve_path(base: Path, value: str | Path | None, field_name: str) -> Path:
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing required path field: {field_name}")
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def object_id_from_entry(entry: dict[str, Any], index: int) -> str:
    return str(entry.get("object_id") or entry.get("id") or entry.get("name") or f"object_{index:02d}")


def default_output_dir(manifest_path: Path, clip_id: str) -> Path:
    return manifest_path.resolve().parent / "captions" / "mucap_input" / clip_id


def load_manifest_objects(manifest_path: Path) -> tuple[str, Path, list[dict[str, Any]]]:
    manifest_path = manifest_path.resolve()
    manifest = load_json(manifest_path)
    clip_id = str(manifest.get("clip_id") or manifest_path.parent.name)
    separated_dir = resolve_path(
        manifest_path.parent,
        manifest.get("output_dir") or "separated_audio",
        "output_dir",
    )
    objects = manifest.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError(f"Manifest must contain a non-empty objects list: {manifest_path}")
    return clip_id, separated_dir, objects


def load_separation_manifest(separated_dir: Path) -> dict[str, Path]:
    summary_path = separated_dir / "separation_manifest.json"
    audio_files: dict[str, Path] = {}
    if not summary_path.exists():
        return audio_files

    summary = load_json(summary_path)
    entries = summary.get("objects")
    if not isinstance(entries, list):
        return audio_files

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        object_id = entry.get("object_id")
        audio_file = entry.get("audio_file")
        if object_id and audio_file:
            audio_files[str(object_id)] = Path(audio_file).resolve()
    return audio_files


def prepare_inputs(
    manifest_path: Path,
    output_dir: Path | None,
    overwrite: bool,
    dry_run: bool,
) -> Path:
    clip_id, separated_dir, objects = load_manifest_objects(manifest_path)
    target_dir = output_dir.resolve() if output_dir is not None else default_output_dir(manifest_path, clip_id)
    manifest_audio = load_separation_manifest(separated_dir)

    records: list[dict[str, Any]] = []
    for index, entry in enumerate(objects, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"objects[{index}] must be an object")
        object_id = object_id_from_entry(entry, index)
        source = manifest_audio.get(object_id) or (separated_dir / object_id / f"{object_id}.wav").resolve()
        if not source.exists():
            raise FileNotFoundError(f"Separated audio does not exist for {object_id}: {source}")

        target = target_dir / f"{object_id}.wav"
        if target.exists() and not overwrite:
            action = "skip"
        else:
            action = "copy"

        print(f"{action.upper()} {object_id}: {source} -> {target}")
        if action == "copy" and not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

        records.append(
            {
                "clip_id": clip_id,
                "object_id": object_id,
                "instrument": entry.get("instrument") or entry.get("class") or entry.get("label") or "",
                "description": entry.get("description") or "",
                "source_audio": str(source),
                "mucap_audio": str(target.resolve()),
            }
        )

    mapping_path = target_dir.parent / "mucap_input_manifest.json"
    print(f"WRITE mapping: {mapping_path}")
    if not dry_run:
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with mapping_path.open("w", encoding="utf-8") as fout:
            json.dump(
                {
                    "clip_id": clip_id,
                    "input_dir": str(target_dir.parent.resolve()),
                    "clip_input_dir": str(target_dir.resolve()),
                    "objects": records,
                },
                fout,
                indent=2,
                ensure_ascii=False,
            )
            fout.write("\n")

    return target_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy separated object wav files into a MuCap batch input directory."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <clip>/captions/mucap_input/<clip_id>.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_dir = prepare_inputs(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(f"MuCap input clip dir: {target_dir}")
    print(f"Run MuCap with --input_dir {target_dir.parent}")


if __name__ == "__main__":
    main()
