import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

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


@dataclass(frozen=True)
class Component:
    label: int
    area: int
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]


@dataclass(frozen=True)
class ObjectSpec:
    object_id: str
    instrument: str
    description: str
    color: Any
    mask_dir: Path


@dataclass(frozen=True)
class MaskStats:
    frame_index: int
    file: str
    time_sec: Optional[float]
    count: int
    bbox: Optional[tuple[int, int, int, int]]
    center: Optional[tuple[float, float]]
    object_id: str = ""
    instrument: str = ""
    description: str = ""
    color: str = ""
    clip_id: str = ""


def natural_frame_index(path: Path) -> int:
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


def frame_time_sec(path: Path, frame_index: int, frame_interval: float | None) -> Optional[float]:
    match = re.search(r"_([0-9]+(?:\.[0-9]+)?)s\.(?:png|jpg|jpeg)$", path.name, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    if frame_interval is not None:
        return (frame_index - 1) * frame_interval
    return None


def image_paths(frame_dir: Path) -> list[Path]:
    paths = [
        path
        for path in frame_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(paths, key=lambda path: (natural_frame_index(path), path.name))


def default_gray_frames_dir(manifest_path: Path) -> Path:
    return manifest_path.resolve().parent / "detections" / "grayscale_reference"


def default_colored_frames_dir(manifest_path: Path) -> Path:
    return manifest_path.resolve().parent / "detections" / "colored_on_black"


def manifest_colored_frames_dir(manifest_path: Path) -> Path | None:
    with manifest_path.open("r", encoding="utf-8") as fin:
        data = json.load(fin)
    if not isinstance(data, dict):
        raise ValueError(f"Manifest root must be a JSON object: {manifest_path}")
    value = data.get("colored_frames_dir")
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = manifest_path.resolve().parent / path
    return path


def match_frame(reference_path: Path, candidates_by_name: dict[str, Path], candidates: list[Path]) -> Path:
    if reference_path.name in candidates_by_name:
        return candidates_by_name[reference_path.name]
    index = natural_frame_index(reference_path)
    matches = [path for path in candidates if natural_frame_index(path) == index and index > 0]
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not match frame for {reference_path.name}")


def resize_rgb_to_shape(image: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    height, width = shape[:2]
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize((width, height), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized.convert("RGB"))


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


def color_label(value: Any) -> str:
    if isinstance(value, str):
        return value
    return ",".join(str(part) for part in parse_color(value))


def color_mask(image: np.ndarray, color: Any, tolerance: int) -> np.ndarray:
    rgb = image[:, :, :3].astype(np.float32)
    target = np.asarray(parse_color(color), dtype=np.float32)
    distance = np.abs(rgb - target).max(axis=2)
    return distance <= tolerance


def black_mask(image: np.ndarray, black_tolerance: int) -> np.ndarray:
    rgb = image[:, :, :3].astype(np.float32)
    return rgb.max(axis=2) <= black_tolerance


def diff_color_mask(
    gray_reference: np.ndarray,
    colored: np.ndarray,
    color: Any,
    tolerance: int,
    diff_threshold: int,
) -> np.ndarray:
    if gray_reference.shape != colored.shape:
        colored = resize_rgb_to_shape(colored, gray_reference.shape)
    rgb_gray = gray_reference[:, :, :3].astype(np.float32)
    rgb_colored = colored[:, :, :3].astype(np.float32)
    changed_mask = np.abs(rgb_colored - rgb_gray).max(axis=2) >= diff_threshold
    return changed_mask & color_mask(colored, color, tolerance)


def measure_mask(mask: np.ndarray) -> tuple[int, Optional[tuple[int, int, int, int]], Optional[tuple[float, float]]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, None, None

    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    center = (float(xs.mean()), float(ys.mean()))
    return len(xs), bbox, center


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, list[Component]] | None:
    try:
        import cv2
    except ImportError:
        return None

    labels_count, labels, stats, centers = cv2.connectedComponentsWithStats(
        mask.astype("uint8"),
        connectivity=8,
    )
    components = []
    for label in range(1, labels_count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        components.append(
            Component(
                label=label,
                area=area,
                bbox=(x, y, x + width - 1, y + height - 1),
                center=(float(centers[label][0]), float(centers[label][1])),
            )
        )
    return labels, components


def bbox_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(0, max(ax1 - bx2, bx1 - ax2))
    dy = max(0, max(ay1 - by2, by1 - ay2))
    return float((dx * dx + dy * dy) ** 0.5)


def filter_main_components(
    mask: np.ndarray,
    min_component_area: int,
    min_component_area_ratio: float,
    component_merge_distance: float,
    max_merged_components: int,
) -> np.ndarray:
    if mask.sum() == 0:
        return mask

    result = connected_components(mask)
    if result is None:
        return mask
    labels, components = result
    if not components:
        return mask

    components = sorted(components, key=lambda component: component.area, reverse=True)
    main = components[0]
    min_area = max(min_component_area, int(round(main.area * min_component_area_ratio)))
    candidates = [component for component in components if component.area >= min_area]
    if not candidates:
        candidates = [main]

    kept: list[Component] = []
    for component in candidates:
        if component.label == main.label or bbox_distance(main.bbox, component.bbox) <= component_merge_distance:
            kept.append(component)
        if max_merged_components > 0 and len(kept) >= max_merged_components:
            break

    if not kept:
        kept = [main]

    kept_labels = {component.label for component in kept}
    return np.isin(labels, list(kept_labels))


def frame_mask_stats(
    frame_dir: Path,
    color: Any,
    tolerance: int,
    frame_interval: float | None,
    object_spec: ObjectSpec | None = None,
    clip_id: str = "",
    min_component_area: int = 80,
    min_component_area_ratio: float = 0.04,
    component_merge_distance: float = 140.0,
    max_merged_components: int = 4,
) -> list[MaskStats]:
    paths = image_paths(frame_dir)
    if not paths:
        raise FileNotFoundError(f"No image frames found in {frame_dir}")

    stats = []
    for idx, path in enumerate(paths, 1):
        frame_index = natural_frame_index(path) or idx
        image = np.asarray(Image.open(path).convert("RGB"))
        mask = color_mask(image, color, tolerance)
        mask = filter_main_components(
            mask,
            min_component_area=min_component_area,
            min_component_area_ratio=min_component_area_ratio,
            component_merge_distance=component_merge_distance,
            max_merged_components=max_merged_components,
        )
        count, bbox, center = measure_mask(mask)
        stats.append(
            MaskStats(
                frame_index=frame_index,
                file=path.name,
                time_sec=frame_time_sec(path, frame_index, frame_interval),
                count=count,
                bbox=bbox,
                center=center,
                object_id=object_spec.object_id if object_spec else "",
                instrument=object_spec.instrument if object_spec else "",
                description=object_spec.description if object_spec else "",
                color=color_label(color),
                clip_id=clip_id,
            )
        )
    return stats


def frame_object_diff_stats(
    gray_frames_dir: Path,
    objects: list[ObjectSpec],
    tolerance: int,
    diff_threshold: int,
    frame_interval: float | None,
    clip_id: str,
    min_component_area: int,
    min_component_area_ratio: float,
    component_merge_distance: float,
    max_merged_components: int,
) -> list[MaskStats]:
    gray_paths = image_paths(gray_frames_dir)
    if not gray_paths:
        raise FileNotFoundError(f"No grayscale reference frames found in {gray_frames_dir}")

    gray_by_name = {path.name: path for path in gray_paths}
    stats = []
    for object_spec in objects:
        object_paths = image_paths(object_spec.mask_dir)
        if not object_paths:
            raise FileNotFoundError(
                f"No separated object frames found for {object_spec.object_id}: "
                f"{object_spec.mask_dir}"
            )

        for idx, object_path in enumerate(object_paths, 1):
            frame_index = natural_frame_index(object_path) or idx
            gray_path = match_frame(object_path, gray_by_name, gray_paths)
            gray_reference = np.asarray(Image.open(gray_path).convert("RGB"))
            object_image = np.asarray(Image.open(object_path).convert("RGB"))
            mask = diff_color_mask(
                gray_reference=gray_reference,
                colored=object_image,
                color=object_spec.color,
                tolerance=tolerance,
                diff_threshold=diff_threshold,
            )
            mask = filter_main_components(
                mask,
                min_component_area=min_component_area,
                min_component_area_ratio=min_component_area_ratio,
                component_merge_distance=component_merge_distance,
                max_merged_components=max_merged_components,
            )
            count, bbox, center = measure_mask(mask)
            stats.append(
                MaskStats(
                    frame_index=frame_index,
                    file=object_path.name,
                    time_sec=frame_time_sec(object_path, frame_index, frame_interval),
                    count=count,
                    bbox=bbox,
                    center=center,
                    object_id=object_spec.object_id,
                    instrument=object_spec.instrument,
                    description=object_spec.description,
                    color=color_label(object_spec.color),
                    clip_id=clip_id,
                )
            )
    return stats


def direct_colored_stats(
    colored_frames_dir: Path,
    objects: list[ObjectSpec],
    tolerance: int,
    black_tolerance: int,
    frame_interval: float | None,
    clip_id: str,
    min_component_area: int,
    min_component_area_ratio: float,
    component_merge_distance: float,
    max_merged_components: int,
) -> list[MaskStats]:
    colored_paths = image_paths(colored_frames_dir)
    if not colored_paths:
        raise FileNotFoundError(f"No colored object frames found in {colored_frames_dir}")

    stats = []
    for idx, colored_path in enumerate(colored_paths, 1):
        frame_index = natural_frame_index(colored_path) or idx
        image = np.asarray(Image.open(colored_path).convert("RGB"))
        non_black = ~black_mask(image, black_tolerance)
        for object_spec in objects:
            mask = non_black & color_mask(image, object_spec.color, tolerance)
            mask = filter_main_components(
                mask,
                min_component_area=min_component_area,
                min_component_area_ratio=min_component_area_ratio,
                component_merge_distance=component_merge_distance,
                max_merged_components=max_merged_components,
            )
            count, bbox, center = measure_mask(mask)
            stats.append(
                MaskStats(
                    frame_index=frame_index,
                    file=colored_path.name,
                    time_sec=frame_time_sec(colored_path, frame_index, frame_interval),
                    count=count,
                    bbox=bbox,
                    center=center,
                    object_id=object_spec.object_id,
                    instrument=object_spec.instrument,
                    description=object_spec.description,
                    color=color_label(object_spec.color),
                    clip_id=clip_id,
                )
            )
    return stats


def resolve_manifest_path(manifest_base: Path, value: str | Path, field_name: str) -> Path:
    if value is None or str(value).strip() == "":
        raise ValueError(f"Manifest is missing required path field: {field_name}")
    path = Path(value)
    if not path.is_absolute():
        path = manifest_base / path
    return path


def load_manifest(manifest_path: Path) -> tuple[str, list[ObjectSpec], Path | None]:
    manifest_path = manifest_path.resolve()
    with manifest_path.open("r", encoding="utf-8") as fin:
        data = json.load(fin)
    if not isinstance(data, dict):
        raise ValueError(f"Manifest root must be a JSON object: {manifest_path}")

    base = manifest_path.parent
    clip_id = str(data.get("clip_id") or base.name)
    output_dir = data.get("output_dir")
    output_path = resolve_manifest_path(base, output_dir, "output_dir") if output_dir else None
    has_colored_frames = bool(data.get("colored_frames_dir"))

    entries = data.get("objects")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest must contain a non-empty objects list")

    objects = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"objects[{index}] must be a JSON object")
        object_id = str(entry.get("object_id") or entry.get("id") or entry.get("name") or f"object_{index:02d}")
        instrument = str(entry.get("instrument") or entry.get("class") or entry.get("label") or object_id)
        description = str(entry.get("description") or instrument)
        color = entry.get("color")
        if color is None:
            raise ValueError(f"objects[{index}] is missing color")
        mask_value = entry.get("mask_dir") or entry.get("frames_dir") or entry.get("frame_dir")
        if mask_value is None and not has_colored_frames:
            raise ValueError(
                f"objects[{index}] is missing mask_dir and manifest has no colored_frames_dir"
            )
        mask_dir = (
            resolve_manifest_path(base, mask_value, f"objects[{index}].mask_dir")
            if mask_value is not None
            else base / "detections" / "masks_by_object" / object_id
        )
        objects.append(
            ObjectSpec(
                object_id=object_id,
                instrument=instrument,
                description=description,
                color=color,
                mask_dir=mask_dir,
            )
        )

    return clip_id, objects, output_path


def load_object_plan(plan_path: Path) -> tuple[str, list[ObjectSpec]]:
    plan_path = plan_path.resolve()
    with plan_path.open("r", encoding="utf-8") as fin:
        data = json.load(fin)
    if not isinstance(data, dict):
        raise ValueError(f"Object plan root must be a JSON object: {plan_path}")

    entries = data.get("objects")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Object plan must contain a non-empty objects list")

    clip_dir = plan_path.parent.parent if plan_path.parent.name == "detections" else plan_path.parent
    objects = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"objects[{index}] must be a JSON object")
        object_id = str(entry.get("object_id") or entry.get("id") or entry.get("name") or f"object_{index:02d}")
        instrument = str(
            entry.get("instrument")
            or entry.get("class")
            or entry.get("label")
            or entry.get("description")
            or object_id
        )
        description = str(entry.get("description") or instrument)
        color = entry.get("color")
        if color is None:
            raise ValueError(f"objects[{index}] must include color")
        objects.append(
            ObjectSpec(
                object_id=object_id,
                instrument=instrument,
                description=description,
                color=color,
                mask_dir=clip_dir / "detections" / "masks_by_object" / object_id,
            )
        )
    return clip_dir.name, objects


def manifest_stats(
    manifest_path: Path,
    tolerance: int,
    diff_threshold: int,
    frame_interval: float | None,
    gray_frames_dir: Path | None,
    min_component_area: int,
    min_component_area_ratio: float,
    component_merge_distance: float,
    max_merged_components: int,
) -> tuple[str, list[MaskStats], Path | None]:
    clip_id, objects, output_path = load_manifest(manifest_path)
    gray_frames_dir = gray_frames_dir or default_gray_frames_dir(manifest_path)
    if gray_frames_dir.exists():
        return (
            clip_id,
            frame_object_diff_stats(
                gray_frames_dir=gray_frames_dir,
                objects=objects,
                tolerance=tolerance,
                diff_threshold=diff_threshold,
                frame_interval=frame_interval,
                clip_id=clip_id,
                min_component_area=min_component_area,
                min_component_area_ratio=min_component_area_ratio,
                component_merge_distance=component_merge_distance,
                max_merged_components=max_merged_components,
            ),
            output_path,
        )

    stats = []
    for object_spec in objects:
        stats.extend(
            frame_mask_stats(
                frame_dir=object_spec.mask_dir,
                color=object_spec.color,
                tolerance=tolerance,
                frame_interval=frame_interval,
                object_spec=object_spec,
                clip_id=clip_id,
                min_component_area=min_component_area,
                min_component_area_ratio=min_component_area_ratio,
                component_merge_distance=component_merge_distance,
                max_merged_components=max_merged_components,
            )
        )
    return clip_id, stats, output_path


def default_manifest_csv(manifest_path: Path, output_path: Path | None, clip_id: str) -> Path:
    if output_path is not None:
        base = output_path.parent if output_path.name == "separated_audio" else output_path
        return base / "coords" / "object_coords.csv"
    return manifest_path.resolve().parent / "coords" / "object_coords.csv"


def print_stats(stats: Iterable[MaskStats]) -> None:
    for stat in stats:
        label = f"{stat.object_id} " if stat.object_id else ""
        if stat.bbox is None or stat.center is None:
            print(f"{label}frame {stat.frame_index:02d}: count=0 bbox=None center=None file={stat.file}")
            continue

        x1, y1, x2, y2 = stat.bbox
        cx, cy = stat.center
        print(
            f"{label}frame {stat.frame_index:02d}: count={stat.count} "
            f"bbox=(x {x1}~{x2}, y {y1}~{y2}) "
            f"center=({cx:.1f}, {cy:.1f}) file={stat.file}"
        )


def row_from_stat(stat: MaskStats, include_object_columns: bool) -> dict[str, Any]:
    row: dict[str, Any] = {}
    if include_object_columns:
        row.update(
            {
                "clip_id": stat.clip_id,
                "object_id": stat.object_id,
                "instrument": stat.instrument,
                "description": stat.description,
                "color": stat.color,
            }
        )
    row.update(
        {
            "frame_index": stat.frame_index,
            "time_sec": "" if stat.time_sec is None else f"{stat.time_sec:.3f}",
            "file": stat.file,
            "count": stat.count,
            "x_min": "",
            "y_min": "",
            "x_max": "",
            "y_max": "",
            "center_x": "",
            "center_y": "",
        }
    )
    if stat.bbox is not None and stat.center is not None:
        row.update(
            {
                "x_min": stat.bbox[0],
                "y_min": stat.bbox[1],
                "x_max": stat.bbox[2],
                "y_max": stat.bbox[3],
                "center_x": f"{stat.center[0]:.3f}",
                "center_y": f"{stat.center[1]:.3f}",
            }
        )
    return row


def write_csv(stats: Iterable[MaskStats], output: Path, include_object_columns: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    base_fields = [
        "frame_index",
        "time_sec",
        "file",
        "count",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "center_x",
        "center_y",
    ]
    object_fields = ["clip_id", "object_id", "instrument", "description", "color"]
    fieldnames = object_fields + base_fields if include_object_columns else base_fields

    with output.open("w", newline="", encoding="utf-8") as fin:
        writer = csv.DictWriter(fin, fieldnames=fieldnames)
        writer.writeheader()
        for stat in stats:
            writer.writerow(row_from_stat(stat, include_object_columns))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-frame bbox and center coordinates from colored object mask frames."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="Manifest generated by create_manifest.py.")
    source.add_argument("--object-plan", type=Path, help="object_plan.json for direct colored-frame coordinate export.")
    source.add_argument("--frame-dir", type=Path, help="Legacy single-object colored frame directory.")
    parser.add_argument("--color", default=None, help="Required with --frame-dir. Supports named colors, #RRGGBB, or R,G,B.")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--tolerance", type=int, default=60)
    parser.add_argument(
        "--colored-frames-dir",
        type=Path,
        default=None,
        help=(
            "All-object GPT colored frames. With --manifest this enables direct coordinate export "
            "without reading per-object mask folders. With --object-plan this is required."
        ),
    )
    parser.add_argument(
        "--black-tolerance",
        type=int,
        default=30,
        help="Pixels with all RGB channels at or below this value are treated as black background.",
    )
    parser.add_argument("--diff-threshold", type=int, default=25)
    parser.add_argument("--min-component-area", type=int, default=80)
    parser.add_argument("--min-component-area-ratio", type=float, default=0.04)
    parser.add_argument("--component-merge-distance", type=float, default=140.0)
    parser.add_argument("--max-merged-components", type=int, default=4)
    parser.add_argument("--gray-frames-dir", type=Path, default=None)
    parser.add_argument(
        "--frame-interval",
        type=float,
        default=None,
        help="Fallback seconds between frames when filenames do not contain timestamps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.manifest is not None:
        clip_id, objects, output_path = load_manifest(args.manifest)
        colored_frames_dir = args.colored_frames_dir or manifest_colored_frames_dir(args.manifest)
        if colored_frames_dir is not None:
            stats = direct_colored_stats(
                colored_frames_dir=colored_frames_dir,
                objects=objects,
                tolerance=args.tolerance,
                black_tolerance=args.black_tolerance,
                frame_interval=args.frame_interval,
                clip_id=clip_id,
                min_component_area=args.min_component_area,
                min_component_area_ratio=args.min_component_area_ratio,
                component_merge_distance=args.component_merge_distance,
                max_merged_components=args.max_merged_components,
            )
        else:
            clip_id, stats, output_path = manifest_stats(
                args.manifest,
                args.tolerance,
                args.diff_threshold,
                args.frame_interval,
                args.gray_frames_dir,
                args.min_component_area,
                args.min_component_area_ratio,
                args.component_merge_distance,
                args.max_merged_components,
            )
        output_csv = args.csv or default_manifest_csv(args.manifest, output_path, clip_id)
        print_stats(stats)
        write_csv(stats, output_csv, include_object_columns=True)
        print(f"Saved CSV: {output_csv}")
        return

    if args.object_plan is not None:
        if args.colored_frames_dir is None:
            raise ValueError("--colored-frames-dir is required with --object-plan")
        clip_id, objects = load_object_plan(args.object_plan)
        stats = direct_colored_stats(
            colored_frames_dir=args.colored_frames_dir,
            objects=objects,
            tolerance=args.tolerance,
            black_tolerance=args.black_tolerance,
            frame_interval=args.frame_interval,
            clip_id=clip_id,
            min_component_area=args.min_component_area,
            min_component_area_ratio=args.min_component_area_ratio,
            component_merge_distance=args.component_merge_distance,
            max_merged_components=args.max_merged_components,
        )
        output_csv = args.csv or args.object_plan.resolve().parent.parent / "coords" / "object_coords.csv"
        print_stats(stats)
        write_csv(stats, output_csv, include_object_columns=True)
        print(f"Saved CSV: {output_csv}")
        return

    if args.color is None:
        raise ValueError("--color is required when using --frame-dir")

    stats = frame_mask_stats(
        args.frame_dir,
        args.color,
        args.tolerance,
        args.frame_interval,
        min_component_area=args.min_component_area,
        min_component_area_ratio=args.min_component_area_ratio,
        component_merge_distance=args.component_merge_distance,
        max_merged_components=args.max_merged_components,
    )
    print_stats(stats)
    if args.csv is not None:
        write_csv(stats, args.csv, include_object_columns=False)
        print(f"Saved CSV: {args.csv}")


if __name__ == "__main__":
    main()
