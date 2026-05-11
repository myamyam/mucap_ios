import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image


@dataclass
class MaskStats:
    frame_index: int
    file: str
    count: int
    bbox: Optional[tuple[int, int, int, int]]
    center: Optional[tuple[float, float]]


def natural_frame_index(path: Path) -> int:
    match = re.search(r"\((\d+)\)\.png$", path.name)
    if match:
        return int(match.group(1))
    match = re.search(r"_(\d+)_\d+\.\d+s\.png$", path.name)
    if match:
        return int(match.group(1))
    return 0


def color_mask(image: np.ndarray, color: str) -> np.ndarray:
    red = image[:, :, 0].astype(np.float32)
    green = image[:, :, 1].astype(np.float32)
    blue = image[:, :, 2].astype(np.float32)

    if color == "red":
        return (red >= 180) & (green <= 110) & (blue <= 110)
    if color == "yellow":
        return (red >= 180) & (green >= 180) & (blue <= 100) & (np.abs(red - green) <= 70)
    if color == "green":
        return (green >= 170) & (red <= 130) & (blue <= 130)
    if color == "blue":
        return (blue >= 180) & (red <= 130) & (green <= 160)

    raise ValueError(f"Unsupported color: {color}")


def measure_mask(mask: np.ndarray) -> tuple[int, Optional[tuple[int, int, int, int]], Optional[tuple[float, float]]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, None, None

    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    center = (float(xs.mean()), float(ys.mean()))
    return len(xs), bbox, center


def frame_mask_stats(frame_dir: Path, color: str) -> list[MaskStats]:
    paths = sorted(frame_dir.glob("*.png"), key=natural_frame_index)
    if not paths:
        raise FileNotFoundError(f"No PNG frames found in {frame_dir}")

    stats = []
    for idx, path in enumerate(paths, 1):
        image = np.asarray(Image.open(path).convert("RGB"))
        count, bbox, center = measure_mask(color_mask(image, color))
        stats.append(
            MaskStats(
                frame_index=idx,
                file=path.name,
                count=count,
                bbox=bbox,
                center=center,
            )
        )
    return stats


def print_stats(stats: Iterable[MaskStats]) -> None:
    for stat in stats:
        if stat.bbox is None or stat.center is None:
            print(f"frame {stat.frame_index:02d}: count=0 bbox=None center=None file={stat.file}")
            continue

        x1, y1, x2, y2 = stat.bbox
        cx, cy = stat.center
        print(
            f"frame {stat.frame_index:02d}: count={stat.count} "
            f"bbox=(x {x1}~{x2}, y {y1}~{y2}) "
            f"center=({cx:.1f}, {cy:.1f}) file={stat.file}"
        )


def write_csv(stats: Iterable[MaskStats], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fin:
        writer = csv.DictWriter(
            fin,
            fieldnames=[
                "frame_index",
                "file",
                "count",
                "x_min",
                "y_min",
                "x_max",
                "y_max",
                "center_x",
                "center_y",
            ],
        )
        writer.writeheader()
        for stat in stats:
            row = {
                "frame_index": stat.frame_index,
                "file": stat.file,
                "count": stat.count,
                "x_min": "",
                "y_min": "",
                "x_max": "",
                "y_max": "",
                "center_x": "",
                "center_y": "",
            }
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
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-dir", required=True, type=Path)
    parser.add_argument("--color", required=True, choices=["red", "yellow", "green", "blue"])
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    stats = frame_mask_stats(args.frame_dir, args.color)
    print_stats(stats)
    if args.csv is not None:
        write_csv(stats, args.csv)
        print(f"Saved CSV: {args.csv}")


if __name__ == "__main__":
    main()
