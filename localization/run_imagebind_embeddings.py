import argparse
import logging
import math
import wave
from pathlib import Path

import cv2
import numpy as np
import torch
import torchaudio
from torchvision import transforms
from torchvision.transforms import functional as F

from feature_adapter import align_depth_tokens_to_vision
from imagebind import ModalityType
from imagebind.models import imagebind_model


DEFAULT_AUDIO_FRAME_SHIFT_MS = 10


def parse_args():
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Export ImageBind patch embeddings with optional clip-axis preservation."
    )
    parser.add_argument("--visual-dir", type=Path, default=root_dir / "clips")
    parser.add_argument("--audio-dir", type=Path, default=root_dir / "audio_data")
    parser.add_argument("--depth-dir", type=Path, default=root_dir / "depth_data")
    parser.add_argument(
        "--output",
        type=Path,
        default=root_dir / "localization" / "imagebind_embeddings.pt",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--clip-duration", type=float, default=2.0)
    parser.add_argument("--clips-per-video", type=int, default=5)
    parser.add_argument(
        "--average-clips",
        action="store_true",
        help="Average clip embeddings back to [B, N, D]. Default keeps [B, S, N, D].",
    )
    return parser.parse_args()


def sorted_files(directory: Path, patterns: tuple[str, ...]) -> list[Path]:
    files = []
    for pattern in patterns:
        files.extend(directory.glob(pattern))
    return sorted(files)


def infer_square_grid_size(num_tokens: int) -> tuple[int, int]:
    grid_size = int(math.sqrt(num_tokens))
    if grid_size * grid_size != num_tokens:
        raise ValueError(f"Cannot infer a square grid from {num_tokens} tokens.")
    return grid_size, grid_size


def get_uniform_clip_timepoints(duration: float, clip_duration: float, clips_per_video: int):
    if clips_per_video <= 0:
        raise ValueError("clips_per_video must be positive.")
    if duration <= 0:
        raise ValueError("Media duration must be positive.")

    max_start = max(duration - clip_duration, 0.0)
    if clips_per_video == 1:
        starts = [max_start * 0.5]
    else:
        starts = [
            max_start * index / (clips_per_video - 1)
            for index in range(clips_per_video)
        ]
    return [(start, min(start + clip_duration, duration)) for start in starts]


def read_video_frames(video_path: Path) -> tuple[list[torch.Tensor], float, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0)
    capture.release()
    if fps <= 0 or not frames:
        raise ValueError(f"Could not read video frames: {video_path}")
    return frames, fps, len(frames) / fps


def sample_frame_indices(
    start: float,
    end: float,
    fps: float,
    frame_count: int,
    num_samples: int,
) -> list[int]:
    if num_samples == 1:
        times = [(start + end) * 0.5]
    else:
        span = max(end - start, 0.0)
        times = [
            start + span * index / (num_samples - 1)
            for index in range(num_samples)
        ]
    return [
        min(max(int(round(time * fps)), 0), frame_count - 1)
        for time in times
    ]


def resize_center_crop(frame: torch.Tensor, size: int = 224) -> torch.Tensor:
    frame = F.resize(frame, size, interpolation=transforms.InterpolationMode.BICUBIC)
    return F.center_crop(frame, [size, size])


def normalize_rgb_video(clip: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor((0.48145466, 0.4578275, 0.40821073), dtype=clip.dtype)
    std = torch.tensor((0.26862954, 0.26130258, 0.27577711), dtype=clip.dtype)
    return (clip - mean[:, None, None, None]) / std[:, None, None, None]


def load_visual_clips(
    video_paths: list[Path],
    device: torch.device,
    clip_duration: float,
    clips_per_video: int,
) -> torch.Tensor:
    video_outputs = []
    for video_path in video_paths:
        source_frames, fps, duration = read_video_frames(video_path)
        clips = []
        for start, end in get_uniform_clip_timepoints(
            duration,
            clip_duration,
            clips_per_video,
        ):
            frames = [
                resize_center_crop(source_frames[frame_index])
                for frame_index in sample_frame_indices(
                    start,
                    end,
                    fps,
                    len(source_frames),
                    2,
                )
            ]
            video_clip = torch.stack(frames, dim=1)
            clips.append(normalize_rgb_video(video_clip))
        video_outputs.append(torch.stack(clips, dim=0))

    return torch.stack(video_outputs, dim=0).to(device)


def waveform2melspec(waveform, sample_rate, num_mel_bins, target_length):
    waveform -= waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sample_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=num_mel_bins,
        dither=0.0,
        frame_length=25,
        frame_shift=DEFAULT_AUDIO_FRAME_SHIFT_MS,
    )
    fbank = fbank.transpose(0, 1)
    n_frames = fbank.size(1)
    p = target_length - n_frames
    if abs(p) / n_frames > 0.2:
        logging.warning(
            "Large gap between audio n_frames(%d) and target_length(%d).",
            n_frames,
            target_length,
        )
    if p > 0:
        fbank = torch.nn.functional.pad(fbank, (0, p), mode="constant", value=0)
    elif p < 0:
        fbank = fbank[:, 0:target_length]
    return fbank.unsqueeze(0)


def load_wav_audio(audio_path: Path) -> tuple[torch.Tensor, int]:
    if audio_path.suffix.lower() != ".wav":
        raise ValueError(
            f"Only WAV audio is supported without torchcodec: {audio_path}"
        )

    with wave.open(str(audio_path), "rb") as wav_file:
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        raw = wav_file.readframes(num_frames)

    if sample_width == 1:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        audio = audio / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
        audio = audio / 2147483648.0
    else:
        raise ValueError(
            f"Unsupported WAV sample width {sample_width} bytes: {audio_path}"
        )

    audio = audio.reshape(-1, num_channels).T
    return torch.from_numpy(audio), sample_rate


def load_audio_clips(
    audio_paths: list[Path],
    device: torch.device,
    clip_duration: float,
    clips_per_video: int,
    num_mel_bins: int = 128,
    target_length: int = 204,
    sample_rate: int = 16000,
    mean: float = -4.268,
    std: float = 9.138,
) -> torch.Tensor:
    audio_outputs = []
    normalize = transforms.Normalize(mean=mean, std=std)
    for audio_path in audio_paths:
        waveform, sr = load_wav_audio(audio_path)
        if sample_rate != sr:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sr,
                new_freq=sample_rate,
            )
        duration = waveform.size(1) / sample_rate
        clips = []
        for start, end in get_uniform_clip_timepoints(
            duration,
            clip_duration,
            clips_per_video,
        ):
            waveform_clip = waveform[
                :,
                int(start * sample_rate) : int(end * sample_rate),
            ]
            clips.append(
                normalize(
                    waveform2melspec(
                        waveform_clip,
                        sample_rate,
                        num_mel_bins,
                        target_length,
                    )
                )
            )
        audio_outputs.append(torch.stack(clips, dim=0))
    return torch.stack(audio_outputs, dim=0).to(device)


def load_depth_clips(
    depth_paths: list[Path],
    device: torch.device,
    clip_duration: float,
    clips_per_video: int,
) -> torch.Tensor:
    depth_outputs = []
    for depth_path in depth_paths:
        source_frames, fps, duration = read_video_frames(depth_path)
        clips = []
        for start, end in get_uniform_clip_timepoints(
            duration,
            clip_duration,
            clips_per_video,
        ):
            frame_index = sample_frame_indices(
                start,
                end,
                fps,
                len(source_frames),
                1,
            )[0]
            depth_frame = source_frames[frame_index].mean(dim=0, keepdim=True)
            depth_frame = resize_center_crop(depth_frame)
            depth_frame = (depth_frame - 0.5) / 0.5
            clips.append(depth_frame)
        depth_outputs.append(torch.stack(clips, dim=0))

    return torch.stack(depth_outputs, dim=0).to(device)


def align_depth_clip_tokens(
    vision_tokens: torch.Tensor,
    depth_tokens: torch.Tensor,
) -> torch.Tensor:
    if vision_tokens.ndim == 4:
        B, S = vision_tokens.shape[:2]
        vision_flat = vision_tokens.reshape(B * S, *vision_tokens.shape[2:])
        depth_flat = depth_tokens.reshape(B * S, *depth_tokens.shape[2:])
        aligned = align_depth_tokens_to_vision(vision_flat, depth_flat)
        return aligned.reshape(B, S, *aligned.shape[1:])

    return align_depth_tokens_to_vision(vision_tokens, depth_tokens)


def main():
    args = parse_args()
    device = torch.device(args.device)

    visual_paths = sorted_files(args.visual_dir, ("*.mp4", "*.mov", "*.mkv", "*.avi", "*.webm"))
    audio_paths = sorted_files(args.audio_dir, ("*.wav", "*.mp3", "*.flac", "*.ogg"))
    depth_paths = sorted_files(args.depth_dir, ("*.mp4", "*.mov", "*.mkv", "*.avi", "*.webm"))

    if not (len(visual_paths) == len(audio_paths) == len(depth_paths)):
        raise ValueError(
            "visual/audio/depth counts must match: "
            f"{len(visual_paths)}, {len(audio_paths)}, {len(depth_paths)}"
        )

    inputs = {
        ModalityType.VISION: load_visual_clips(
            visual_paths,
            device,
            clip_duration=args.clip_duration,
            clips_per_video=args.clips_per_video,
        ),
        ModalityType.AUDIO: load_audio_clips(
            audio_paths,
            device,
            clip_duration=args.clip_duration,
            clips_per_video=args.clips_per_video,
        ),
        ModalityType.DEPTH: load_depth_clips(
            depth_paths,
            device,
            clip_duration=args.clip_duration,
            clips_per_video=args.clips_per_video,
        ),
    }

    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    with torch.inference_mode():
        tokens = model(
            inputs,
            return_patch_embeddings=True,
            keep_clip_dim=not args.average_clips,
        )

    tokens[ModalityType.DEPTH] = align_depth_clip_tokens(
        tokens[ModalityType.VISION],
        tokens[ModalityType.DEPTH],
    )

    stems = [f"sample_{index + 1}" for index in range(len(visual_paths))]
    payload = {
        "stems": stems,
        "paths": {
            "clips": [str(path) for path in visual_paths],
            "audio": [str(path) for path in audio_paths],
            "depth": [str(path) for path in depth_paths],
        },
        "tokens": {key: value.cpu() for key, value in tokens.items()},
        "token_grids": {
            "vision": infer_square_grid_size(tokens[ModalityType.VISION].shape[-2]),
            "depth_aligned_to_vision": infer_square_grid_size(
                tokens[ModalityType.DEPTH].shape[-2]
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)

    print(f"saved: {args.output}")
    print(f"stems: {stems}")
    for key, value in payload["tokens"].items():
        print(f"{key}: {tuple(value.shape)}")


if __name__ == "__main__":
    main()
