import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

COLOR_PALETTE = {
    "red": (255, 45, 45),
    "yellow": (255, 220, 30),
    "green": (35, 220, 80),
    "blue": (55, 145, 255),
    "magenta": (220, 80, 255),
    "orange": (255, 140, 45),
    "cyan": (40, 220, 220),
    "purple": (123, 44, 255),
    "teal": (0, 184, 148),
    "white": (255, 255, 255),  # legacy object_plan support
}


@dataclass(frozen=True)
class ObjectMetadata:
    instrument: str | None = None
    description: str | None = None
    color: str | None = None


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def relative_to_clip(path: Path, clip_dir: Path) -> str:
    return Path(path).resolve().relative_to(clip_dir.resolve()).as_posix()


def relpath(path: Path, start: Path) -> str:
    return Path(os.path.relpath(path.resolve(), start.resolve())).as_posix()


def image_paths(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: (frame_sort_key(path), path.name),
    )


def frame_sort_key(path: Path) -> int:
    patterns = [
        r"\((\d+)\)\.(?:png|jpg|jpeg)$",
        r"_(\d+)(?:_[0-9.]+s)?\.(?:png|jpg|jpeg)$",
        r"frame[_-]?(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, path.name, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return 0


def find_clip_dirs(assets_root: Path, explicit_clip_dirs: list[str] | None) -> list[Path]:
    if explicit_clip_dirs:
        return [resolve_path(path) for path in explicit_clip_dirs]

    assets_root = resolve_path(assets_root)
    if not assets_root.exists():
        raise FileNotFoundError(f"Assets root does not exist: {assets_root}")

    clip_dirs = sorted([path for path in assets_root.iterdir() if path.is_dir()])
    if not clip_dirs:
        raise FileNotFoundError(f"No clip directories found in {assets_root}")
    return clip_dirs


def first_existing_image_dir(candidates: list[Path], field_name: str) -> Path:
    for candidate in candidates:
        if candidate.exists() and image_paths(candidate):
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {field_name}. Searched: {searched}")


def find_audio(clip_dir: Path, clip_id: str) -> Path:
    exact = clip_dir / "audio" / f"{clip_id}.wav"
    if exact.exists():
        return exact

    candidates = sorted((clip_dir / "audio").glob("*.wav")) if (clip_dir / "audio").exists() else []
    candidates.extend(sorted(clip_dir.glob("*.wav")))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"Could not find wav audio for {clip_id}. Expected audio/{clip_id}.wav or audio/*.wav"
    )


def find_visual_dir(clip_dir: Path) -> Path:
    frame_dirs = sorted(
        [path for path in clip_dir.iterdir() if path.is_dir() and path.name.startswith("frames_")]
    )
    candidates = [
        clip_dir / "frames_0_5s",
        clip_dir / "frames",
        clip_dir / "raw_visual1_frames",
        *frame_dirs,
    ]
    return first_existing_image_dir(candidates, "visual frame directory")


def find_mask_root(clip_dir: Path) -> Path:
    candidates = [
        clip_dir / "detections" / "masks_by_object",
        clip_dir / "masks_by_object",
        clip_dir / "raw_visual1_frames_coloredbyobject",
    ]
    for candidate in candidates:
        if candidate.exists() and any(path.is_dir() for path in candidate.iterdir()):
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find object mask root. Searched: {searched}")


def infer_instrument(object_id: str) -> str:
    name = object_id.strip().lower()
    name = re.sub(r"[-\s]+", "_", name)
    name = re.sub(r"_(?:\d+|[ivx]+)$", "", name)
    name = re.sub(r"_(?:left|right|front|back|rear)$", "", name)
    return name or object_id


def load_metadata(clip_dir: Path, metadata_name: str) -> dict[str, ObjectMetadata]:
    candidates = [
        clip_dir / metadata_name,
        clip_dir / "detections" / metadata_name,
        clip_dir / "detections" / "object_metadata.json",
        clip_dir / "object_metadata.json",
    ]
    metadata_path = next((path for path in candidates if path.exists()), None)
    if metadata_path is None:
        return {}

    with metadata_path.open("r", encoding="utf-8") as fin:
        data = json.load(fin)

    entries = data.get("objects", data) if isinstance(data, dict) else data
    if not isinstance(entries, list):
        raise ValueError(f"Object metadata must be a list or contain an objects list: {metadata_path}")

    metadata: dict[str, ObjectMetadata] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        object_id = entry.get("object_id") or entry.get("id") or entry.get("name")
        if not object_id:
            continue
        metadata[str(object_id)] = ObjectMetadata(
            instrument=entry.get("instrument") or entry.get("class") or entry.get("label"),
            description=entry.get("description"),
            color=entry.get("color"),
        )
    return metadata


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)


def sample_indices(length: int, max_count: int) -> list[int]:
    if length <= 0:
        return []
    if max_count <= 0 or length <= max_count:
        return list(range(length))
    return sorted(set(np.linspace(0, length - 1, max_count).round().astype(int).tolist()))


def estimate_color(
    visual_dir: Path,
    object_dir: Path,
    max_frames: int,
    diff_threshold: int,
    min_changed_pixels: int,
) -> str:
    visual_frames = image_paths(visual_dir)
    object_frames = image_paths(object_dir)
    if not visual_frames:
        raise FileNotFoundError(f"No visual frames found in {visual_dir}")
    if not object_frames:
        raise FileNotFoundError(f"No object frames found in {object_dir}")

    visual_by_name = {path.name: path for path in visual_frames}
    pairs = [(visual_by_name[path.name], path) for path in object_frames if path.name in visual_by_name]
    if not pairs:
        count = min(len(visual_frames), len(object_frames))
        pairs = list(zip(visual_frames[:count], object_frames[:count], strict=False))

    changed_pixels = []
    for index in sample_indices(len(pairs), max_frames):
        visual_path, object_path = pairs[index]
        visual = load_rgb(visual_path)
        colored = load_rgb(object_path)
        if visual.shape != colored.shape:
            raise ValueError(
                f"Frame size mismatch while estimating color: {visual_path} vs {object_path}"
            )

        diff = np.abs(colored - visual).max(axis=2)
        changed = colored[diff >= diff_threshold]
        if changed.size:
            changed_pixels.append(changed)

    if not changed_pixels:
        raise ValueError(
            f"Could not find changed object pixels in {object_dir}. "
            "Check that object frames are colored versions of the visual frames."
        )

    pixels = np.concatenate(changed_pixels, axis=0).astype(np.float32)
    if len(pixels) < min_changed_pixels:
        raise ValueError(
            f"Only found {len(pixels)} changed pixels in {object_dir}; "
            f"minimum is {min_changed_pixels}"
        )

    best_color = None
    best_score = -1
    for color_name, rgb in COLOR_PALETTE.items():
        target = np.asarray(rgb, dtype=np.float32)
        distance = np.abs(pixels - target).max(axis=1)
        score = int((distance <= 80).sum())
        if score > best_score:
            best_score = score
            best_color = color_name

    if best_color is None or best_score == 0:
        mean = pixels.mean(axis=0)
        best_color = min(
            COLOR_PALETTE,
            key=lambda name: float(np.abs(mean - np.asarray(COLOR_PALETTE[name])).max()),
        )
    return best_color


def object_dirs(mask_root: Path) -> list[Path]:
    dirs = sorted([path for path in mask_root.iterdir() if path.is_dir()])
    if not dirs:
        raise FileNotFoundError(f"No object directories found in {mask_root}")
    return dirs


def build_manifest(
    clip_dir: Path,
    outputs_root: Path,
    metadata_name: str,
    max_color_frames: int,
    diff_threshold: int,
    min_changed_pixels: int,
) -> dict[str, Any]:
    clip_dir = clip_dir.resolve()
    clip_id = clip_dir.name
    outputs_root = resolve_path(outputs_root)

    audio = find_audio(clip_dir, clip_id)
    visual_dir = find_visual_dir(clip_dir)
    mask_root = find_mask_root(clip_dir)
    metadata = load_metadata(clip_dir, metadata_name)

    objects = []
    for object_dir in object_dirs(mask_root):
        object_id = object_dir.name
        entry_metadata = metadata.get(object_id, ObjectMetadata())
        instrument = entry_metadata.instrument or infer_instrument(object_id)
        description = entry_metadata.description or instrument
        color = entry_metadata.color or estimate_color(
            visual_dir=visual_dir,
            object_dir=object_dir,
            max_frames=max_color_frames,
            diff_threshold=diff_threshold,
            min_changed_pixels=min_changed_pixels,
        )

        objects.append(
            {
                "object_id": object_id,
                "instrument": instrument,
                "description": description,
                "color": color,
                "mask_dir": relative_to_clip(object_dir, clip_dir),
            }
        )

    output_dir = outputs_root / clip_id / "separated_audio"
    return {
        "clip_id": clip_id,
        "audio": relative_to_clip(audio, clip_dir),
        "visual_dir": relative_to_clip(visual_dir, clip_dir),
        "output_dir": relpath(output_dir, clip_dir),
        "objects": objects,
    }


def write_manifest(clip_dir: Path, manifest: dict[str, Any], manifest_name: str, overwrite: bool) -> Path:
    output_path = clip_dir / manifest_name
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Manifest already exists: {output_path}. Use --overwrite to replace it.")

    with output_path.open("w", encoding="utf-8") as fout:
        json.dump(manifest, fout, indent=2, ensure_ascii=False)
        fout.write("\n")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create manifest.json files for SAM-Audio visual object separation."
    )
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=Path("outputs/video_assets"),
        help="Root containing one folder per clip.",
    )
    parser.add_argument(
        "--clip-dir",
        action="append",
        default=None,
        help="Specific clip directory. Can be passed multiple times. Overrides --assets-root scanning.",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs/video_assets"),
        help="Root for final outputs; manifests point separated audio here.",
    )
    parser.add_argument("--manifest-name", default="manifest.json")
    parser.add_argument("--metadata-name", default="object_metadata.json")
    parser.add_argument("--max-color-frames", type=int, default=6)
    parser.add_argument("--diff-threshold", type=int, default=35)
    parser.add_argument("--min-changed-pixels", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clip_dirs = find_clip_dirs(args.assets_root, args.clip_dir)

    written = 0
    for clip_dir in clip_dirs:
        manifest = build_manifest(
            clip_dir=clip_dir,
            outputs_root=args.outputs_root,
            metadata_name=args.metadata_name,
            max_color_frames=args.max_color_frames,
            diff_threshold=args.diff_threshold,
            min_changed_pixels=args.min_changed_pixels,
        )

        if args.dry_run:
            print(json.dumps(manifest, indent=2, ensure_ascii=False))
            continue

        manifest_path = write_manifest(
            clip_dir=clip_dir,
            manifest=manifest,
            manifest_name=args.manifest_name,
            overwrite=args.overwrite,
        )
        written += 1
        print(f"Saved {manifest_path}")

    if not args.dry_run:
        print(f"Done. Saved {written} manifest file(s).")


if __name__ == "__main__":
    main()
