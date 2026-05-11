import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont


COLOR_RGB = {
    "red": (255, 45, 45),
    "yellow": (255, 220, 30),
    "green": (35, 220, 80),
    "blue": (55, 145, 255),
    "white": (255, 255, 255),
}

PALETTE = [
    (255, 45, 45),
    (255, 220, 30),
    (35, 220, 80),
    (55, 145, 255),
    (220, 80, 255),
    (255, 140, 45),
    (40, 220, 220),
    (255, 255, 255),
]


@dataclass
class ObjectBox:
    object_name: str
    color: str
    time_sec: Optional[float]
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    center_x: float
    center_y: float


def natural_frame_index(path: Path) -> int:
    match = re.search(r"\((\d+)\)\.png$", path.name)
    if match:
        return int(match.group(1))
    match = re.search(r"_(\d+)_\d+\.\d+s\.png$", path.name)
    if match:
        return int(match.group(1))
    return 0


def frame_index_map(frame_paths: list[Path]) -> dict[Path, int]:
    parsed = [natural_frame_index(path) for path in frame_paths]
    if all(index > 0 for index in parsed) and len(set(parsed)) == len(parsed):
        return dict(zip(frame_paths, parsed, strict=True))
    return {path: idx for idx, path in enumerate(frame_paths, 1)}


def parse_float(value: str) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def load_boxes(csv_path: Path) -> dict[int, list[ObjectBox]]:
    boxes_by_frame: dict[int, list[ObjectBox]] = defaultdict(list)
    with csv_path.open("r", newline="", encoding="utf-8-sig") as fin:
        reader = csv.DictReader(fin)
        required = {
            "frame_index",
            "object",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
            "center_x",
            "center_y",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")

        for row in reader:
            if row.get("is_sounding", "True").lower() == "false":
                continue

            coords = [parse_float(row[name]) for name in ["x_min", "y_min", "x_max", "y_max", "center_x", "center_y"]]
            if any(value is None for value in coords):
                continue

            frame_index = int(float(row["frame_index"]))
            boxes_by_frame[frame_index].append(
                ObjectBox(
                    object_name=row["object"],
                    color=row.get("color") or "white",
                    time_sec=parse_float(row.get("time_sec", "")),
                    x_min=coords[0],
                    y_min=coords[1],
                    x_max=coords[2],
                    y_max=coords[3],
                    center_x=coords[4],
                    center_y=coords[5],
                )
            )
    return boxes_by_frame


def resolve_color(color_name: str, object_name: str) -> tuple[int, int, int]:
    normalized = color_name.lower().strip()
    if normalized in COLOR_RGB:
        return COLOR_RGB[normalized]
    if re.fullmatch(r"#[0-9a-fA-F]{6}", normalized):
        return tuple(int(normalized[i : i + 2], 16) for i in (1, 3, 5))
    return PALETTE[abs(hash(object_name)) % len(PALETTE)]


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color: tuple[int, int, int], font: ImageFont.ImageFont) -> None:
    left, top = xy
    bbox = draw.textbbox((left, top), text, font=font)
    pad = 4
    bg_box = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    draw.rectangle(bg_box, fill=(0, 0, 0))
    draw.text((left, top), text, fill=color, font=font)


def draw_center(draw: ImageDraw.ImageDraw, x: float, y: float, color: tuple[int, int, int], radius: int) -> None:
    cx = int(round(x))
    cy = int(round(y))
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color, outline=(0, 0, 0), width=2)
    draw.line((cx - radius * 2, cy, cx + radius * 2, cy), fill=(0, 0, 0), width=1)
    draw.line((cx, cy - radius * 2, cx, cy + radius * 2), fill=(0, 0, 0), width=1)


def draw_overlay(
    image_path: Path,
    boxes: list[ObjectBox],
    output_path: Path,
    line_width: int,
    center_radius: int,
    scale_x: float,
    scale_y: float,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for box in boxes:
        color = resolve_color(box.color, box.object_name)
        rect = (
            int(round(box.x_min * scale_x)),
            int(round(box.y_min * scale_y)),
            int(round(box.x_max * scale_x)),
            int(round(box.y_max * scale_y)),
        )
        draw.rectangle(rect, outline=color, width=line_width)
        draw_center(draw, box.center_x * scale_x, box.center_y * scale_y, color, center_radius)

        time_label = f" {box.time_sec:.1f}s" if box.time_sec is not None else ""
        label = f"{box.object_name}{time_label}"
        label_y = max(0, rect[1] - 16)
        draw_label(draw, (rect[0], label_y), label, color, font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_bbox_overlay"))
    parser.add_argument(
        "--coord-frames-dir",
        type=Path,
        default=None,
        help="Frame folder whose image size matches CSV coordinates. If set, coordinates are scaled to --frames-dir size.",
    )
    parser.add_argument("--line-width", type=int, default=4)
    parser.add_argument("--center-radius", type=int, default=6)
    args = parser.parse_args()

    boxes_by_frame = load_boxes(args.csv)
    frame_paths = sorted(args.frames_dir.glob("*.png"), key=natural_frame_index)
    if not frame_paths:
        raise FileNotFoundError(f"No PNG frames found in {args.frames_dir}")

    target_size = Image.open(frame_paths[0]).size
    coord_size = target_size
    if args.coord_frames_dir is not None:
        coord_paths = sorted(args.coord_frames_dir.glob("*.png"), key=natural_frame_index)
        if not coord_paths:
            raise FileNotFoundError(f"No PNG frames found in {args.coord_frames_dir}")
        coord_size = Image.open(coord_paths[0]).size

    scale_x = target_size[0] / coord_size[0]
    scale_y = target_size[1] / coord_size[1]
    print(
        f"Coordinate scale: source={coord_size[0]}x{coord_size[1]} "
        f"target={target_size[0]}x{target_size[1]} "
        f"scale=({scale_x:.6f}, {scale_y:.6f})"
    )

    written = 0
    frame_indices = frame_index_map(frame_paths)
    for image_path in frame_paths:
        frame_index = frame_indices[image_path]
        boxes = boxes_by_frame.get(frame_index, [])
        output_path = args.output_dir / f"{image_path.stem}_overlay.png"
        draw_overlay(
            image_path=image_path,
            boxes=boxes,
            output_path=output_path,
            line_width=args.line_width,
            center_radius=args.center_radius,
            scale_x=scale_x,
            scale_y=scale_y,
        )
        written += 1
        print(f"Saved {output_path} ({len(boxes)} boxes)")

    print(f"Done. Saved {written} overlay images to {args.output_dir}")


if __name__ == "__main__":
    main()
