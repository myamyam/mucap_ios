import argparse
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FFMPEG_BIN = ROOT / "tools" / "ffmpeg" / "bin"
if FFMPEG_BIN.exists():
    os.add_dll_directory(str(FFMPEG_BIN))

import torch
import torchaudio
import torchvision

from sam_audio import SAMAudio, SAMAudioProcessor


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


def red_highlight_mask(frames: torch.Tensor, threshold: int, margin: float, dilate: int) -> torch.Tensor:
    rgb = frames.float()
    red = rgb[:, 0]
    green = rgb[:, 1]
    blue = rgb[:, 2]

    mask = (red >= threshold) & (red >= green * margin) & (red >= blue * margin)
    mask = mask.unsqueeze(1).float()

    if dilate > 1:
        if dilate % 2 == 0:
            raise ValueError("--dilate must be an odd number")
        mask = torch.nn.functional.max_pool2d(mask, kernel_size=dilate, stride=1, padding=dilate // 2)

    return mask.bool()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default="audio_data/raw_audio1.wav")
    parser.add_argument("--frame-dir", default="raw_visual1_frames_coloredbyobject/violin1")
    parser.add_argument("--description", default="violin")
    parser.add_argument("--model-id", default="facebook/sam-audio-large-tv")
    parser.add_argument("--output-dir", default="outputs/raw_audio1_violin1_visual")
    parser.add_argument("--reranking-candidates", type=int, default=1)
    parser.add_argument("--red-threshold", type=int, default=120)
    parser.add_argument("--red-margin", type=float, default=1.35)
    parser.add_argument("--dilate", type=int, default=17)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio = Path(args.audio)
    if not audio.is_absolute():
        audio = ROOT / audio

    frame_dir = Path(args.frame_dir)
    if not frame_dir.is_absolute():
        frame_dir = ROOT / frame_dir

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = load_frames(frame_dir)
    mask = red_highlight_mask(
        frames,
        threshold=args.red_threshold,
        margin=args.red_margin,
        dilate=args.dilate,
    )

    model = SAMAudio.from_pretrained(args.model_id).to(device).eval()
    processor = SAMAudioProcessor.from_pretrained(args.model_id)

    masked_videos = processor.mask_videos([frames], [mask])
    batch = processor(
        audios=[str(audio)],
        descriptions=[args.description],
        masked_videos=masked_videos,
    ).to(device)

    with torch.inference_mode():
        result = model.separate(batch, reranking_candidates=args.reranking_candidates)

    sample_rate = processor.audio_sampling_rate
    torchaudio.save(
        output_dir / "violin1.wav",
        result.target[0].detach().cpu().unsqueeze(0),
        sample_rate,
    )
    torchaudio.save(
        output_dir / "residual.wav",
        result.residual[0].detach().cpu().unsqueeze(0),
        sample_rate,
    )

    print(f"Saved {output_dir / 'violin1.wav'}")
    print(f"Saved {output_dir / 'residual.wav'}")


if __name__ == "__main__":
    main()
