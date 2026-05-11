import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FFMPEG_BIN = ROOT / "tools" / "ffmpeg" / "bin"
if FFMPEG_BIN.exists():
    os.add_dll_directory(str(FFMPEG_BIN))

import torch
import torchaudio
import torchvision

from sam_audio import SAMAudio, SAMAudioProcessor


OBJECT_DESCRIPTIONS = {
    "cello": "cello",
    "violin1": "violin",
    "violin2": "violin",
    "violin3": "violin",
}


def frame_index(path: Path) -> int:
    match = re.search(r"\((\d+)\)\.png$", path.name)
    if match:
        return int(match.group(1))
    match = re.search(r"_(\d+)_\d+\.\d+s\.png$", path.name)
    if match:
        return int(match.group(1))
    return 0


def load_frames(frame_dir: Path) -> torch.Tensor:
    paths = sorted(frame_dir.glob("*.png"), key=frame_index)
    if not paths:
        raise FileNotFoundError(f"No PNG frames found in {frame_dir}")

    frames = []
    for path in paths:
        image = torchvision.io.read_image(str(path), mode=torchvision.io.ImageReadMode.RGB)
        frames.append(image)
    return torch.stack(frames, dim=0)


def sample_frames(frames: torch.Tensor, max_frames: int) -> torch.Tensor:
    if max_frames <= 0 or frames.size(0) <= max_frames:
        return frames
    idxs = torch.linspace(0, frames.size(0) - 1, max_frames).round().long()
    return frames[idxs]


def color_mask(frames: torch.Tensor, object_name: str, threshold: int, margin: float, dilate: int) -> torch.Tensor:
    rgb = frames.float()
    red = rgb[:, 0]
    green = rgb[:, 1]
    blue = rgb[:, 2]

    if object_name == "cello":
        mask = (blue >= max(threshold, 180)) & (red <= 130) & (green <= 160)
    elif object_name == "violin2":
        mask = (
            (red >= max(threshold, 180))
            & (green >= max(threshold, 180))
            & (blue <= 100)
            & ((red - green).abs() <= 70)
        )
    elif object_name == "violin3":
        mask = (green >= max(threshold, 170)) & (red <= 130) & (blue <= 130)
    else:
        mask = (red >= max(threshold, 180)) & (green <= 110) & (blue <= 110)

    mask = mask.unsqueeze(1).float()
    if dilate > 1:
        if dilate % 2 == 0:
            raise ValueError("--dilate must be an odd number")
        mask = torch.nn.functional.max_pool2d(mask, kernel_size=dilate, stride=1, padding=dilate // 2)
    return mask.bool()


def print_mask_stats(object_name: str, mask: torch.Tensor) -> None:
    coords = torch.nonzero(mask[:, 0], as_tuple=False)
    if coords.numel() == 0:
        print(f"{object_name} mask: count=0 bbox=None center=None")
        return

    y = coords[:, 1]
    x = coords[:, 2]
    bbox = (int(x.min()), int(y.min()), int(x.max()), int(y.max()))
    center = (float(x.float().mean()), float(y.float().mean()))
    print(
        f"{object_name} mask: count={coords.size(0)} "
        f"bbox=(x {bbox[0]}~{bbox[2]}, y {bbox[1]}~{bbox[3]}) "
        f"center=({center[0]:.1f}, {center[1]:.1f})"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default="audio_data/raw_audio1.wav")
    parser.add_argument("--visual-dir", default="raw_visual1_frames_colored")
    parser.add_argument("--object-root", default="raw_visual1_frames_coloredbyobject")
    parser.add_argument("--objects", nargs="+", default=["cello", "violin1", "violin2", "violin3"])
    parser.add_argument("--model-id", default="facebook/sam-audio-large-tv")
    parser.add_argument("--output-dir", default="outputs/raw_audio1_visual_objects")
    parser.add_argument("--reranking-candidates", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--threshold", type=int, default=120)
    parser.add_argument("--margin", type=float, default=1.25)
    parser.add_argument("--dilate", type=int, default=17)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    audio = Path(args.audio)
    if not audio.is_absolute():
        audio = ROOT / audio

    visual_dir = Path(args.visual_dir)
    if not visual_dir.is_absolute():
        visual_dir = ROOT / visual_dir

    object_root = Path(args.object_root)
    if not object_root.is_absolute():
        object_root = ROOT / object_root

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    visual_frames = sample_frames(load_frames(visual_dir), args.max_frames)

    model = SAMAudio.from_pretrained(args.model_id).to(device).eval()
    processor = SAMAudioProcessor.from_pretrained(args.model_id)
    sample_rate = processor.audio_sampling_rate

    for object_name in args.objects:
        mask_source_dir = object_root / object_name
        mask_source_frames = sample_frames(load_frames(mask_source_dir), args.max_frames)
        if mask_source_frames.shape != visual_frames.shape:
            raise ValueError(
                f"{object_name} frames shape {tuple(mask_source_frames.shape)} does not match "
                f"visual frames shape {tuple(visual_frames.shape)}"
            )

        mask = color_mask(
            mask_source_frames,
            object_name=object_name,
            threshold=args.threshold,
            margin=args.margin,
            dilate=args.dilate,
        )
        print_mask_stats(object_name, mask)

        masked_videos = processor.mask_videos([visual_frames], [mask])
        batch = processor(
            audios=[str(audio)],
            descriptions=[OBJECT_DESCRIPTIONS.get(object_name, object_name)],
            masked_videos=masked_videos,
        ).to(device)

        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if args.amp and device.type == "cuda"
            else torch.autocast(device_type="cpu", enabled=False)
        )
        with torch.inference_mode(), amp_context:
            result = model.separate(batch, reranking_candidates=args.reranking_candidates)

        object_dir = output_dir / object_name
        object_dir.mkdir(parents=True, exist_ok=True)
        torchaudio.save(
            object_dir / f"{object_name}.wav",
            result.target[0].detach().cpu().unsqueeze(0),
            sample_rate,
        )
        torchaudio.save(
            object_dir / "residual.wav",
            result.residual[0].detach().cpu().unsqueeze(0),
            sample_rate,
        )
        print(f"Saved {object_dir / f'{object_name}.wav'}")
        print(f"Saved {object_dir / 'residual.wav'}")


if __name__ == "__main__":
    main()
