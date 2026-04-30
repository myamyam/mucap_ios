import argparse
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args():
    root_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Export OCA sounding maps as heatmap PNGs.")
    parser.add_argument(
        "--input",
        type=Path,
        default=root_dir / "localization" / "loss_oca" / "oca_sounding_maps_affinity.pt",
        help="Sounding-map .pt file from run_sounding_map_affinity.py or run_sounding_map_consistency.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save heatmap PNGs. Defaults to <input_stem>_heatmaps.",
    )
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument(
        "--video-overlay",
        action="store_true",
        help="Also export heatmap-overlaid MP4 videos when maps are [B, S, H, W].",
    )
    parser.add_argument(
        "--video-output-dir",
        type=Path,
        default=None,
        help="Directory to save overlay MP4s. Defaults to <output-dir>_videos.",
    )
    parser.add_argument("--alpha", type=float, default=0.45)
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


def default_output_dir(input_path: Path, mode: str) -> Path:
    if mode in {"affinity", "consistency"}:
        return input_path.parent / f"oca_result_{mode}"
    input_stem = input_path.with_suffix("")
    return input_stem.parent / f"{input_stem.name}_heatmaps"


def clip_index_for_frame(
    frame_index: int,
    frame_count: int,
    num_clips: int,
) -> int:
    if num_clips <= 1 or frame_count <= 1:
        return 0
    position = frame_index / (frame_count - 1)
    return int(round(position * (num_clips - 1)))


def resolve_video_paths(payload: dict, stems: list[str]) -> list[Path]:
    paths = payload.get("paths") or {}
    clip_paths = paths.get("clips") or paths.get("vision") or paths.get("visual")
    if clip_paths is not None:
        return [Path(path) for path in clip_paths]
    return [Path("clips") / f"{stem}.mp4" for stem in stems]


def export_overlay_videos(
    maps: np.ndarray,
    payload: dict,
    stems: list[str],
    mode: str,
    output_dir: Path,
    alpha: float,
) -> None:
    if maps.ndim != 4:
        raise ValueError("--video-overlay requires sounding_maps with shape [B, S, H, W].")

    video_paths = resolve_video_paths(payload, stems)
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample_index, (stem, video_path) in enumerate(zip(stems, video_paths)):
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or width <= 0 or height <= 0 or frame_count <= 0:
            capture.release()
            raise ValueError(f"Could not read video metadata: {video_path}")

        output_path = output_dir / f"{stem}_{mode}_overlay.mp4"
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise ValueError(f"Could not open video writer: {output_path}")

        clip_maps = maps[sample_index]
        heatmaps = [
            to_heatmap(clip_map, max(width, height))
            for clip_map in clip_maps
        ]
        heatmaps = [
            cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
            for heatmap in heatmaps
        ]

        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            clip_index = clip_index_for_frame(frame_index, frame_count, len(heatmaps))
            overlay = cv2.addWeighted(frame, 1.0 - alpha, heatmaps[clip_index], alpha, 0)
            writer.write(overlay)
            frame_index += 1

        capture.release()
        writer.release()
        print(f"saved overlay video: {output_path}")


def main():
    args = parse_args()
    payload = torch.load(args.input, map_location="cpu")
    maps = payload["sounding_maps"].numpy()
    stems = payload["stems"]
    mode = payload.get("mode", args.input.stem)

    output_dir = args.output_dir or default_output_dir(args.input, mode)
    output_dir.mkdir(parents=True, exist_ok=True)

    if maps.ndim == 3:
        for index, stem in enumerate(stems):
            heatmap = to_heatmap(maps[index], args.size)
            output_path = output_dir / f"{stem}_{mode}.png"
            cv2.imwrite(str(output_path), heatmap)
    elif maps.ndim == 4:
        for sample_index, stem in enumerate(stems):
            sample_dir = output_dir / stem
            sample_dir.mkdir(parents=True, exist_ok=True)
            for clip_index in range(maps.shape[1]):
                heatmap = to_heatmap(maps[sample_index, clip_index], args.size)
                output_path = sample_dir / f"{stem}_{mode}_clip{clip_index + 1:03d}.png"
                cv2.imwrite(str(output_path), heatmap)
    else:
        raise ValueError("sounding_maps must be [B, H, W] or [B, S, H, W].")

    if args.video_overlay:
        video_output_dir = args.video_output_dir or output_dir.parent / f"{output_dir.name}_videos"
        export_overlay_videos(
            maps=maps,
            payload=payload,
            stems=stems,
            mode=mode,
            output_dir=video_output_dir,
            alpha=args.alpha,
        )

    print(f"input: {args.input}")
    print(f"mode: {mode}")
    print(f"maps: {tuple(maps.shape)}")
    print(f"saved heatmaps: {output_dir}")


if __name__ == "__main__":
    main()
