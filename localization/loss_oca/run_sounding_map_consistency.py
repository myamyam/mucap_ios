import argparse
import math
from pathlib import Path

import torch

from oca_sounding_map_depth_consistency import (
    build_soft_masks,
    compute_audio_visual_scores,
    compute_depth_prior,
)


def parse_args():
    root_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Run depth-consistency OCA sounding map from ImageBind embeddings."
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=root_dir / "localization" / "imagebind_embeddings.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root_dir / "localization" / "loss_oca" / "oca_sounding_maps_consistency.pt",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tau-a", type=float, default=0.07)
    parser.add_argument("--tau-d", type=float, default=0.07)
    parser.add_argument("--lambda-a", type=float, default=1.0)
    parser.add_argument("--lambda-d", type=float, default=0.2)
    parser.add_argument("--alpha-p", type=float, default=0.4)
    parser.add_argument("--alpha-n", type=float, default=0.4)
    parser.add_argument("--omega", type=float, default=0.1)
    parser.add_argument("--audio-reduction", choices=["attn", "mean", "max"], default="attn")
    return parser.parse_args()


def infer_square_grid_size(num_tokens: int) -> tuple[int, int]:
    grid_size = int(math.sqrt(num_tokens))
    if grid_size * grid_size != num_tokens:
        raise ValueError(f"Cannot infer a square grid from {num_tokens} tokens.")
    return grid_size, grid_size


def infer_clip_shape(named_tensors: dict[str, torch.Tensor]) -> tuple[int, int] | None:
    clip_shape = None
    for name, tensor in named_tensors.items():
        if tensor.ndim not in {3, 4}:
            raise ValueError(f"{name}_tokens must be [B, N, D] or [B, S, N, D].")
        if tensor.ndim == 4:
            shape = tensor.shape[:2]
            if clip_shape is None:
                clip_shape = shape
            elif shape != clip_shape:
                raise ValueError(
                    f"{name}_tokens has clip shape {tuple(shape)}, "
                    f"expected {tuple(clip_shape)}."
                )
    return clip_shape


def flatten_clip_tokens(
    tokens: torch.Tensor,
    clip_shape: tuple[int, int] | None,
    name: str,
) -> torch.Tensor:
    if clip_shape is None:
        if tokens.ndim != 3:
            raise ValueError(f"{name}_tokens must be [B, N, D].")
        return tokens

    B, S = clip_shape
    if tokens.ndim == 4:
        if tokens.shape[:2] != clip_shape:
            raise ValueError(
                f"{name}_tokens has clip shape {tuple(tokens.shape[:2])}, "
                f"expected {clip_shape}."
            )
        return tokens.reshape(B * S, *tokens.shape[2:])

    if tokens.ndim == 3:
        if tokens.shape[0] != B:
            raise ValueError(
                f"{name}_tokens batch size {tokens.shape[0]} does not match {B}."
            )
        return tokens.unsqueeze(1).expand(-1, S, -1, -1).reshape(B * S, *tokens.shape[1:])

    raise ValueError(f"{name}_tokens must be [B, N, D] or [B, S, N, D].")


def restore_clip_outputs(
    tensor: torch.Tensor,
    clip_shape: tuple[int, int] | None,
) -> torch.Tensor:
    if clip_shape is None:
        return tensor
    B, S = clip_shape
    return tensor.reshape(B, S, *tensor.shape[1:])


def main():
    args = parse_args()
    device = torch.device(args.device)
    embeddings = torch.load(args.embeddings, map_location="cpu")

    vision_tokens = embeddings["tokens"]["vision"].to(device)
    audio_tokens = embeddings["tokens"]["audio"].to(device)
    depth_tokens = embeddings["tokens"]["depth"].to(device)
    input_token_shapes = {
        "vision": tuple(vision_tokens.shape),
        "audio": tuple(audio_tokens.shape),
        "depth": tuple(depth_tokens.shape),
    }
    clip_shape = infer_clip_shape(
        {
            "vision": vision_tokens,
            "audio": audio_tokens,
            "depth": depth_tokens,
        }
    )
    vision_tokens = flatten_clip_tokens(vision_tokens, clip_shape, "vision")
    audio_tokens = flatten_clip_tokens(audio_tokens, clip_shape, "audio")
    depth_tokens = flatten_clip_tokens(depth_tokens, clip_shape, "depth")

    with torch.inference_mode():
        audio_scores = compute_audio_visual_scores(
            vision_tokens=vision_tokens,
            audio_tokens=audio_tokens,
            tau_a=args.tau_a,
            reduction=args.audio_reduction,
        )
        depth_scores = compute_depth_prior(
            vision_tokens=vision_tokens,
            depth_tokens=depth_tokens,
            tau_d=args.tau_d,
            mode="aligned",
        )
        fused_scores, foreground_masks, background_masks = build_soft_masks(
            audio_scores=audio_scores,
            depth_scores=depth_scores,
            lambda_a=args.lambda_a,
            lambda_d=args.lambda_d,
            alpha_p=args.alpha_p,
            alpha_n=args.alpha_n,
            omega=args.omega,
        )

    grid_size = infer_square_grid_size(foreground_masks.shape[1])
    sounding_maps = foreground_masks.reshape(foreground_masks.shape[0], *grid_size)
    sounding_maps = restore_clip_outputs(sounding_maps, clip_shape)
    audio_scores = restore_clip_outputs(audio_scores, clip_shape)
    depth_scores = restore_clip_outputs(depth_scores, clip_shape)
    fused_scores = restore_clip_outputs(fused_scores, clip_shape)
    foreground_masks = restore_clip_outputs(foreground_masks, clip_shape)
    background_masks = restore_clip_outputs(background_masks, clip_shape)

    payload = {
        "mode": "consistency",
        "stems": embeddings["stems"],
        "paths": embeddings.get("paths"),
        "grid_size": grid_size,
        "clip_shape": tuple(clip_shape) if clip_shape is not None else None,
        "input_token_shapes": input_token_shapes,
        "audio_scores": audio_scores.cpu(),
        "depth_scores": depth_scores.cpu(),
        "fused_scores": fused_scores.cpu(),
        "foreground_masks": foreground_masks.cpu(),
        "background_masks": background_masks.cpu(),
        "sounding_maps": sounding_maps.cpu(),
        "params": {
            "tau_a": args.tau_a,
            "tau_d": args.tau_d,
            "lambda_a": args.lambda_a,
            "lambda_d": args.lambda_d,
            "alpha_p": args.alpha_p,
            "alpha_n": args.alpha_n,
            "omega": args.omega,
            "audio_reduction": args.audio_reduction,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)

    print(f"saved: {args.output}")
    print(f"stems: {payload['stems']}")
    print(f"clip_shape: {payload['clip_shape']}")
    print(f"sounding_maps: {tuple(sounding_maps.shape)}")


if __name__ == "__main__":
    main()
