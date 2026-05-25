import argparse
import json
import os
import re
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FFMPEG_BIN = ROOT / "tools" / "ffmpeg" / "bin"
if FFMPEG_BIN.exists():
    os.add_dll_directory(str(FFMPEG_BIN))

import torch
import torchaudio
import torchvision

from sam_audio import SAMAudio, SAMAudioProcessor


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

NAMED_COLORS = {
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
class ObjectSpec:
    object_id: str
    instrument: str
    description: str
    mask_dir: Path | None
    color: Any | None


@dataclass(frozen=True)
class SeparationConfig:
    manifest_path: Path
    manifest_base: Path
    clip_id: str
    audio: Path
    visual_dir: Path
    colored_frames_dir: Path | None
    output_dir: Path
    objects: list[ObjectSpec]


def frame_index(path: Path) -> int:
    patterns = [
        r"\((\d+)\)\.(?:png|jpg|jpeg)$",
        r"_(\d+)(?:_[0-9.]+s)?\.(?:png|jpg|jpeg)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, path.name, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return 0


def image_paths(frame_dir: Path) -> list[Path]:
    paths = [
        path
        for path in frame_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(paths, key=lambda path: (frame_index(path), path.name))


def load_frames(frame_dir: Path) -> torch.Tensor:
    paths = image_paths(frame_dir)
    if not paths:
        raise FileNotFoundError(f"No image frames found in {frame_dir}")

    frames = []
    for path in paths:
        image = torchvision.io.read_image(
            str(path),
            mode=torchvision.io.ImageReadMode.RGB,
        )
        frames.append(image)
    return torch.stack(frames, dim=0)


def sample_frames(frames: torch.Tensor, max_frames: int) -> torch.Tensor:
    if max_frames <= 0 or frames.size(0) <= max_frames:
        return frames
    idxs = torch.linspace(0, frames.size(0) - 1, max_frames).round().long()
    return frames[idxs]


def resize_frames_max_side(frames: torch.Tensor, max_side: int) -> torch.Tensor:
    if max_side <= 0:
        return frames

    height, width = frames.shape[-2:]
    current_max = max(height, width)
    if current_max <= max_side:
        return frames

    scale = max_side / current_max
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))
    resized = torch.nn.functional.interpolate(
        frames.float(),
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.clamp(0, 255).to(dtype=frames.dtype)


def resolve_path(base: Path, value: str | Path | None, field_name: str) -> Path:
    if value is None or str(value).strip() == "":
        raise ValueError(f"Manifest is missing required path field: {field_name}")

    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path


def find_manifest(run_dir: Path) -> Path:
    candidates = [
        run_dir / "manifest.json",
        run_dir / "detections" / "manifest.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find manifest.json. Searched: {searched}")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fin:
        data = json.load(fin)
    if not isinstance(data, dict):
        raise ValueError(f"Manifest root must be a JSON object: {path}")
    return data


def object_id_from_entry(entry: dict[str, Any], index: int) -> str:
    value = entry.get("object_id") or entry.get("id") or entry.get("name")
    if value:
        return str(value)
    return f"object_{index:02d}"


def load_config(manifest_path: Path, output_dir_override: str | None) -> SeparationConfig:
    manifest_path = manifest_path.resolve()
    data = load_json(manifest_path)
    base = manifest_path.parent

    visual_value = (
        data.get("visual_dir")
        or data.get("frames_dir")
        or data.get("colored_frames_dir")
    )
    colored_frames_value = data.get("colored_frames_dir")
    output_value = output_dir_override or data.get("output_dir") or "separated_audio"

    object_entries = data.get("objects")
    if not isinstance(object_entries, list) or not object_entries:
        raise ValueError("Manifest must contain a non-empty objects list")

    objects = []
    for index, entry in enumerate(object_entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"objects[{index}] must be a JSON object")

        object_id = object_id_from_entry(entry, index)
        instrument = str(
            entry.get("instrument")
            or entry.get("class")
            or entry.get("label")
            or entry.get("description")
            or object_id
        )
        description = str(entry.get("description") or instrument)
        mask_value = entry.get("mask_dir") or entry.get("frames_dir") or entry.get("frame_dir")
        color = entry.get("color")
        if mask_value is None and color is None:
            raise ValueError(
                f"objects[{index}] must include color when using manifest-level colored_frames_dir"
            )

        objects.append(
            ObjectSpec(
                object_id=object_id,
                instrument=instrument,
                description=description,
                mask_dir=resolve_path(base, mask_value, f"objects[{index}].mask_dir") if mask_value else None,
                color=color,
            )
        )

    return SeparationConfig(
        manifest_path=manifest_path,
        manifest_base=base,
        clip_id=str(data.get("clip_id") or manifest_path.parent.name),
        audio=resolve_path(base, data.get("audio") or data.get("audio_path"), "audio"),
        visual_dir=resolve_path(base, visual_value, "visual_dir"),
        colored_frames_dir=(
            resolve_path(base, colored_frames_value, "colored_frames_dir")
            if colored_frames_value
            else None
        ),
        output_dir=resolve_path(base, output_value, "output_dir"),
        objects=objects,
    )


def parse_color(value: Any) -> tuple[int, int, int]:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in NAMED_COLORS:
            return NAMED_COLORS[normalized]
        if re.fullmatch(r"#[0-9a-f]{6}", normalized):
            return tuple(int(normalized[i : i + 2], 16) for i in (1, 3, 5))
        parts = [part.strip() for part in normalized.split(",")]
        if len(parts) == 3 and all(part.isdigit() for part in parts):
            return tuple(int(part) for part in parts)

    if isinstance(value, list) and len(value) == 3:
        return tuple(int(part) for part in value)

    raise ValueError(f"Unsupported color value: {value!r}")


def dilate_mask(mask: torch.Tensor, dilate: int) -> torch.Tensor:
    mask = mask.unsqueeze(1).float()
    if dilate > 1:
        if dilate % 2 == 0:
            raise ValueError("--dilate must be an odd number")
        mask = torch.nn.functional.max_pool2d(
            mask,
            kernel_size=dilate,
            stride=1,
            padding=dilate // 2,
        )
    return mask.bool()


def color_mask(
    frames: torch.Tensor,
    color: Any | None,
    tolerance: int,
    non_background_threshold: int,
    dilate: int,
) -> torch.Tensor:
    rgb = frames.float()
    if color is None:
        mask = rgb.max(dim=1).values >= non_background_threshold
        return dilate_mask(mask, dilate)

    target = torch.tensor(parse_color(color), dtype=torch.float32)[:, None, None]
    diff = (rgb - target).abs()
    mask = diff.max(dim=1).values <= tolerance
    return dilate_mask(mask, dilate)


def print_mask_stats(object_spec: ObjectSpec, mask: torch.Tensor) -> None:
    coords = torch.nonzero(mask[:, 0], as_tuple=False)
    if coords.numel() == 0:
        print(f"{object_spec.object_id} mask: count=0 bbox=None center=None")
        return

    y = coords[:, 1]
    x = coords[:, 2]
    bbox = (int(x.min()), int(y.min()), int(x.max()), int(y.max()))
    center = (float(x.float().mean()), float(y.float().mean()))
    print(
        f"{object_spec.object_id} mask: count={coords.size(0)} "
        f"bbox=(x {bbox[0]}~{bbox[2]}, y {bbox[1]}~{bbox[3]}) "
        f"center=({center[0]:.1f}, {center[1]:.1f}) "
        f"description={object_spec.description!r}"
    )


def validate_inputs(config: SeparationConfig, max_objects: int) -> None:
    if not config.audio.exists():
        raise FileNotFoundError(f"Audio file does not exist: {config.audio}")
    if not config.visual_dir.exists():
        raise FileNotFoundError(f"Visual frame directory does not exist: {config.visual_dir}")
    if config.colored_frames_dir is not None and not config.colored_frames_dir.exists():
        raise FileNotFoundError(f"Colored frame directory does not exist: {config.colored_frames_dir}")
    if max_objects > 0 and len(config.objects) > max_objects:
        raise ValueError(
            f"Manifest contains {len(config.objects)} objects, "
            f"but --max-objects is {max_objects}"
        )
    for object_spec in config.objects:
        if object_spec.mask_dir is None:
            if config.colored_frames_dir is None:
                raise ValueError(
                    f"{object_spec.object_id} has no mask_dir and manifest has no colored_frames_dir"
                )
            continue
        if not object_spec.mask_dir.exists():
            raise FileNotFoundError(
                f"Mask frame directory does not exist for {object_spec.object_id}: "
                f"{object_spec.mask_dir}"
            )


def write_summary(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "separation_manifest.json"
    with summary_path.open("w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2, ensure_ascii=False)
        fout.write("\n")
    print(f"Saved summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Separate audio for visual objects described by a manifest.json file."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="Path to manifest.json.")
    source.add_argument(
        "--run-dir",
        type=Path,
        help="Run directory containing manifest.json or detections/manifest.json.",
    )
    parser.add_argument("--model-id", default="facebook/sam-audio-large-tv")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override manifest output_dir. Relative paths resolve from the manifest folder.",
    )
    parser.add_argument("--reranking-candidates", type=int, default=1)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Use all frames by default. Set a positive number to sample frames.",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=8,
        help="Guardrail for expected object count. Set 0 to disable.",
    )
    parser.add_argument("--mask-tolerance", type=int, default=60)
    parser.add_argument("--non-background-threshold", type=int, default=30)
    parser.add_argument("--dilate", type=int, default=17)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device for SAM-Audio separation.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=0,
        help="Resize visual/mask frames for separation so the longest side is this size.",
    )
    parser.add_argument("--save-residual", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest or find_manifest(args.run_dir)
    config = load_config(manifest_path, args.output_dir)
    validate_inputs(config, args.max_objects)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    visual_frames_all = load_frames(config.visual_dir)
    colored_frames_all = None
    if config.colored_frames_dir is not None:
        colored_frames_all = load_frames(config.colored_frames_dir)
        if colored_frames_all.shape != visual_frames_all.shape:
            raise ValueError(
                f"Colored frames shape {tuple(colored_frames_all.shape)} does not match "
                f"visual frames shape {tuple(visual_frames_all.shape)}"
            )

    visual_frames = sample_frames(visual_frames_all, args.max_frames)
    visual_frames = resize_frames_max_side(visual_frames, args.max_side)
    if args.max_side > 0:
        print(f"Separation visual frame size: {tuple(visual_frames.shape[-2:])}")

    model = SAMAudio.from_pretrained(args.model_id).to(device).eval()
    processor = SAMAudioProcessor.from_pretrained(args.model_id)
    sample_rate = processor.audio_sampling_rate

    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary_objects = []

    for object_spec in config.objects:
        if object_spec.mask_dir is not None:
            mask_source_frames_all = load_frames(object_spec.mask_dir)
            mask_source_label = str(object_spec.mask_dir)
        elif colored_frames_all is not None:
            mask_source_frames_all = colored_frames_all
            mask_source_label = str(config.colored_frames_dir)
        else:
            raise ValueError(f"No mask source is available for {object_spec.object_id}")

        if mask_source_frames_all.shape != visual_frames_all.shape:
            raise ValueError(
                f"{object_spec.object_id} frames shape "
                f"{tuple(mask_source_frames_all.shape)} does not match visual frames shape "
                f"{tuple(visual_frames_all.shape)}"
            )

        mask_source_frames = sample_frames(mask_source_frames_all, args.max_frames)
        mask_source_frames = resize_frames_max_side(mask_source_frames, args.max_side)
        mask = color_mask(
            mask_source_frames,
            color=object_spec.color,
            tolerance=args.mask_tolerance,
            non_background_threshold=args.non_background_threshold,
            dilate=args.dilate,
        )
        print_mask_stats(object_spec, mask)

        masked_videos = processor.mask_videos([visual_frames], [mask])
        batch = processor(
            audios=[str(config.audio)],
            descriptions=[object_spec.description],
            masked_videos=masked_videos,
        ).to(device)

        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if args.amp and device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), amp_context:
            result = model.separate(
                batch,
                reranking_candidates=args.reranking_candidates,
            )

        object_dir = config.output_dir / object_spec.object_id
        object_dir.mkdir(parents=True, exist_ok=True)
        target_path = object_dir / f"{object_spec.object_id}.wav"
        torchaudio.save(
            target_path,
            result.target[0].detach().cpu().unsqueeze(0),
            sample_rate,
        )
        print(f"Saved {target_path}")

        residual_path = None
        if args.save_residual:
            residual_path = object_dir / "residual.wav"
            torchaudio.save(
                residual_path,
                result.residual[0].detach().cpu().unsqueeze(0),
                sample_rate,
            )
            print(f"Saved {residual_path}")

        summary_objects.append(
            {
                "object_id": object_spec.object_id,
                "instrument": object_spec.instrument,
                "description": object_spec.description,
                "color": object_spec.color,
                "mask_dir": str(object_spec.mask_dir) if object_spec.mask_dir else None,
                "mask_source": mask_source_label,
                "audio_file": str(target_path),
                "residual_file": str(residual_path) if residual_path else None,
            }
        )

    write_summary(
        config.output_dir,
        {
            "clip_id": config.clip_id,
            "manifest": str(config.manifest_path),
            "audio": str(config.audio),
            "visual_dir": str(config.visual_dir),
            "colored_frames_dir": str(config.colored_frames_dir) if config.colored_frames_dir else None,
            "model_id": args.model_id,
            "sample_rate": sample_rate,
            "max_frames": args.max_frames,
            "objects": summary_objects,
        },
    )


if __name__ == "__main__":
    main()
