import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (ROOT / path).resolve()


def hsv_histogram(frame_bgr: np.ndarray, resize_width: int, resize_height: int) -> np.ndarray:
    resized = cv2.resize(frame_bgr, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
    return hist


def detect_scene_cuts(
    input_video: Path,
    threshold: float,
    min_gap_sec: float,
    frame_step: int,
    resize_width: int,
    resize_height: int,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, int(frame_step))

    previous_hist: np.ndarray | None = None
    last_cut_time = -float("inf")
    cuts: list[dict[str, float]] = []
    distances: list[float] = []

    frame_index = 0
    processed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % frame_step != 0:
            frame_index += 1
            continue

        time_sec = frame_index / fps
        hist = hsv_histogram(frame, resize_width, resize_height)
        if previous_hist is not None:
            distance = float(cv2.compareHist(previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
            distances.append(distance)
            if distance >= threshold and time_sec - last_cut_time >= min_gap_sec:
                cuts.append({"time_sec": round(time_sec, 4), "distance": round(distance, 6)})
                last_cut_time = time_sec
        previous_hist = hist
        processed += 1
        frame_index += 1

    cap.release()
    return {
        "input_video": str(input_video),
        "method": "hsv_hs_histogram_bhattacharyya",
        "threshold": threshold,
        "min_gap_sec": min_gap_sec,
        "frame_step": frame_step,
        "resize": [resize_width, resize_height],
        "fps": fps,
        "frame_count": frame_count,
        "processed_frames": processed,
        "cuts": cuts,
        "distance_summary": {
            "count": len(distances),
            "max": max(distances) if distances else 0.0,
            "mean": float(np.mean(distances)) if distances else 0.0,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect hard scene cuts using HSV histogram differences.")
    parser.add_argument("--input", type=Path, required=True, help="Input MP4 video.")
    parser.add_argument("--output", type=Path, required=True, help="Output scene_cuts.json.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-gap", type=float, default=0.4)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--resize-width", type=int, default=160)
    parser.add_argument("--resize-height", type=int, default=90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_video = resolve_project_path(args.input)
    output_path = resolve_project_path(args.output)
    result = detect_scene_cuts(
        input_video=input_video,
        threshold=args.threshold,
        min_gap_sec=args.min_gap,
        frame_step=args.frame_step,
        resize_width=args.resize_width,
        resize_height=args.resize_height,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        json.dump(result, fout, indent=2)
        fout.write("\n")

    print(f"Input video: {input_video}")
    print(f"Output: {output_path}")
    print(f"Detected cuts: {len(result['cuts'])}")
    if result["cuts"]:
        print("Cut times: " + ", ".join(f"{cut['time_sec']:.2f}s" for cut in result["cuts"]))


if __name__ == "__main__":
    main()
