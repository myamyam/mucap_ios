import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

COLOR_RGB = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "magenta": (255, 0, 255),
    "cyan": (0, 255, 255),
    "orange": (255, 128, 0),
    "purple": (128, 0, 255),
    "white": (255, 255, 255),  # legacy object_plan support
}


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = (base or ROOT) / value
    return value


def frame_index(path: Path) -> int:
    patterns = [
        r"\((\d+)\)\.(?:png|jpg|jpeg|webp)$",
        r"_(\d+)(?:_[0-9.]+s)?\.(?:png|jpg|jpeg|webp)$",
        r"frame[_-]?(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, path.name, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return 0


def image_paths(directory: Path) -> list[Path]:
    paths = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(paths, key=lambda path: (frame_index(path), path.name))


def parse_color(value: Any) -> tuple[int, int, int]:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in COLOR_RGB:
            return COLOR_RGB[normalized]
        if re.fullmatch(r"#[0-9a-f]{6}", normalized):
            return tuple(int(normalized[i : i + 2], 16) for i in (1, 3, 5))
        parts = [part.strip() for part in normalized.split(",")]
        if len(parts) == 3 and all(part.isdigit() for part in parts):
            return tuple(int(part) for part in parts)
    if isinstance(value, list) and len(value) == 3:
        return tuple(int(part) for part in value)
    raise ValueError(f"Unsupported color: {value!r}")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fin:
        data = json.load(fin)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def default_object_plan_path(frames_dir: Path) -> Path:
    return frames_dir.parent / "detections" / "object_plan.json"


def default_gray_frames_dir(frames_dir: Path) -> Path:
    return frames_dir.parent / "detections" / "grayscale_reference"


def default_colored_on_gray_dir(frames_dir: Path) -> Path:
    return frames_dir.parent / "detections" / "colored_on_gray"


def load_object_plan(plan_path: Path) -> list[dict[str, Any]]:
    data = load_json(plan_path)
    entries = data.get("objects")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Object plan must contain a non-empty objects list: {plan_path}")
    objects = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"objects[{index}] must be an object")
        object_id = entry.get("object_id") or entry.get("id") or entry.get("name")
        color = entry.get("color")
        if not object_id or color is None:
            raise ValueError(f"objects[{index}] must include object_id and color")
        objects.append(
            {
                "object_id": str(object_id),
                "instrument": str(entry.get("instrument") or entry.get("class") or entry.get("label") or object_id),
                "description": str(entry.get("description") or entry.get("instrument") or object_id),
                "color": color,
                "rgb": parse_color(color),
            }
        )
    return objects


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)


def resize_rgb_to_shape(image: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    height, width = shape[:2]
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize((width, height), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized.convert("RGB"), dtype=np.int16)


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.astype(np.uint8), mode="RGB").save(path)


def make_object_image(
    gray_reference: np.ndarray,
    colored: np.ndarray,
    target_rgb: tuple[int, int, int],
    color_tolerance: int,
    diff_threshold: int,
    background_mode: str,
) -> tuple[np.ndarray, int]:
    diff = np.abs(colored - gray_reference).max(axis=2)
    changed_mask = diff >= diff_threshold

    target = np.asarray(target_rgb, dtype=np.int16)
    color_distance = np.abs(colored - target).max(axis=2)
    color_mask = color_distance <= color_tolerance
    mask = changed_mask & color_mask

    if background_mode == "original":
        output = gray_reference.copy()
    elif background_mode == "black":
        output = np.zeros_like(gray_reference)
    else:
        raise ValueError(f"Unsupported background mode: {background_mode}")

    output[mask] = target
    return output, int(mask.sum())


def match_colored_frame(original_path: Path, colored_by_name: dict[str, Path], colored_paths: list[Path]) -> Path:
    if original_path.name in colored_by_name:
        return colored_by_name[original_path.name]
    index = frame_index(original_path)
    matches = [path for path in colored_paths if frame_index(path) == index and index > 0]
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not match colored frame for {original_path.name}")


def write_metadata(output_root: Path, objects: list[dict[str, Any]]) -> Path:
    payload = {
        "objects": [
            {
                "object_id": obj["object_id"],
                "instrument": obj["instrument"],
                "description": obj["description"],
                "color": obj["color"],
            }
            for obj in objects
        ]
    }
    output_path = output_root.parent / "object_metadata.json"
    with output_path.open("w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2, ensure_ascii=False)
        fout.write("\n")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split GPT all-object colored frames into one object-colored frame sequence per object."
    )
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument(
        "--colored-on-gray-dir",
        dest="colored_on_gray_dir",
        type=Path,
        default=None,
        help="Defaults to <clip>/detections/colored_on_gray.",
    )
    parser.add_argument(
        "--gray-frames-dir",
        type=Path,
        default=None,
        help="Defaults to <clip>/detections/grayscale_reference.",
    )
    parser.add_argument(
        "--object-plan",
        type=Path,
        default=None,
        help="Defaults to <clip>/detections/object_plan.json.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Defaults to <clip>/detections/masks_by_object.",
    )
    parser.add_argument("--color-tolerance", type=int, default=60)
    parser.add_argument("--diff-threshold", type=int, default=25)
    parser.add_argument("--background-mode", choices=["original", "black"], default="black")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--write-metadata", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames_dir = resolve_path(args.frames_dir)
    clip_dir = frames_dir.parent
    gray_frames_dir = resolve_path(args.gray_frames_dir) if args.gray_frames_dir else default_gray_frames_dir(frames_dir)
    colored_on_gray_dir = (
        resolve_path(args.colored_on_gray_dir)
        if args.colored_on_gray_dir
        else default_colored_on_gray_dir(frames_dir)
    )
    plan_path = resolve_path(args.object_plan) if args.object_plan else default_object_plan_path(frames_dir)
    output_root = resolve_path(args.output_root) if args.output_root else clip_dir / "detections" / "masks_by_object"

    objects = load_object_plan(plan_path)
    original_paths = image_paths(frames_dir)
    gray_paths = image_paths(gray_frames_dir)
    colored_paths = image_paths(colored_on_gray_dir)
    if not original_paths:
        raise FileNotFoundError(f"No original frames found in {frames_dir}")
    if not gray_paths:
        raise FileNotFoundError(f"No grayscale reference frames found in {gray_frames_dir}")
    if not colored_paths:
        raise FileNotFoundError(f"No colored-on-gray frames found in {colored_on_gray_dir}")

    gray_by_name = {path.name: path for path in gray_paths}
    colored_by_name = {path.name: path for path in colored_paths}
    print(f"Frames: {len(original_paths)}")
    print(f"Objects: {len(objects)}")
    print(f"Gray reference: {gray_frames_dir}")
    print(f"Colored on gray: {colored_on_gray_dir}")
    print(f"Output root: {output_root}")

    for frame_count, original_path in enumerate(original_paths, start=1):
        gray_path = match_colored_frame(original_path, gray_by_name, gray_paths)
        colored_path = match_colored_frame(original_path, colored_by_name, colored_paths)
        original = load_rgb(original_path)
        gray_reference = load_rgb(gray_path)
        colored = load_rgb(colored_path)
        if original.shape != gray_reference.shape:
            print(
                "Resizing gray reference to original frame size: "
                f"{gray_path.name} {gray_reference.shape[1]}x{gray_reference.shape[0]} -> "
                f"{original.shape[1]}x{original.shape[0]}"
            )
            gray_reference = resize_rgb_to_shape(gray_reference, original.shape)
        if original.shape != colored.shape:
            print(
                "Resizing colored frame to original frame size: "
                f"{colored_path.name} {colored.shape[1]}x{colored.shape[0]} -> "
                f"{original.shape[1]}x{original.shape[0]}"
            )
            colored = resize_rgb_to_shape(colored, original.shape)

        for obj in objects:
            output_path = output_root / obj["object_id"] / original_path.with_suffix(".png").name
            if args.skip_existing and output_path.exists():
                continue
            object_image, count = make_object_image(
                gray_reference=gray_reference,
                colored=colored,
                target_rgb=obj["rgb"],
                color_tolerance=args.color_tolerance,
                diff_threshold=args.diff_threshold,
                background_mode=args.background_mode,
            )
            save_rgb(output_path, object_image)
            print(
                f"[{frame_count}/{len(original_paths)}] "
                f"{obj['object_id']} {original_path.name}: pixels={count} -> {output_path}"
            )

    if args.write_metadata:
        metadata_path = write_metadata(output_root, objects)
        print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
