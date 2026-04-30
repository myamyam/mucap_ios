import argparse
from pathlib import Path

import torch

from ori_reference_maps_depth import (
    build_ori_reference_maps,
    pool_audio_tokens_as_single_reference,
)


def parse_args():
    root_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Run depth-aware ORI reference maps from ImageBind embeddings."
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=root_dir / "localization" / "imagebind_embeddings.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root_dir / "localization" / "loss_ori" / "ori_reference_maps.pt",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--depth-weight", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    embeddings = torch.load(args.embeddings, map_location="cpu")

    vision_tokens = embeddings["tokens"]["vision"].to(device)
    audio_tokens = embeddings["tokens"]["audio"].to(device)
    depth_tokens = embeddings["tokens"]["depth"].to(device)
    audio_refs = pool_audio_tokens_as_single_reference(audio_tokens)

    with torch.inference_mode():
        sr_scores, sr_map, stats = build_ori_reference_maps(
            vision_tokens=vision_tokens,
            audio_refs=audio_refs,
            background_refs=None,
            depth_tokens=depth_tokens,
            depth_weight=args.depth_weight,
            temperature=args.temperature,
        )

    payload = {
        "stems": embeddings["stems"],
        "sr_scores": sr_scores.cpu(),
        "sr_map": sr_map.cpu(),
        "ref_features": stats["ref_features"].cpu(),
        "fused_tokens": stats["fused_tokens"].cpu(),
        "params": {
            "depth_weight": args.depth_weight,
            "temperature": args.temperature,
            "reference": "pooled_audio_tokens",
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)

    print(f"saved: {args.output}")
    print(f"stems: {payload['stems']}")
    print(f"sr_map: {tuple(sr_map.shape)}")


if __name__ == "__main__":
    main()
