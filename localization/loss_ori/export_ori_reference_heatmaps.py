import argparse
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args():
    root_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Export ORI reference maps as heatmap PNGs.")
    parser.add_argument(
        "--input",
        type=Path,
        default=root_dir / "localization" / "loss_ori" / "ori_reference_maps.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root_dir / "localization" / "loss_ori" / "ori_result",
    )
    parser.add_argument("--size", type=int, default=224)
    return parser.parse_args()


def normalize_map(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.nanmin(x)
    denom = np.nanmax(x)
    if denom > 0:
        x = x / denom
    return x


def to_heatmap(x: np.ndarray, size: int) -> np.ndarray:
    x = normalize_map(x)
    x = cv2.resize(x, (size, size), interpolation=cv2.INTER_CUBIC)
    x = (x * 255).clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(x, cv2.COLORMAP_JET)


def main():
    args = parse_args()
    payload = torch.load(args.input, map_location="cpu")
    sr_map = payload["sr_map"].numpy()
    stems = payload["stems"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for batch_index, stem in enumerate(stems):
        for ref_index in range(sr_map.shape[1]):
            heatmap = to_heatmap(sr_map[batch_index, ref_index], args.size)
            output_path = args.output_dir / f"{stem}_ori_ref{ref_index + 1}.png"
            cv2.imwrite(str(output_path), heatmap)

    print(f"input: {args.input}")
    print(f"sr_map: {tuple(sr_map.shape)}")
    print(f"saved heatmaps: {args.output_dir}")


if __name__ == "__main__":
    main()
