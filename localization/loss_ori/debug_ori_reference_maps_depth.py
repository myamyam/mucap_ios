import torch

from ori_reference_maps_depth import build_ori_reference_maps


def print_tensor_stats(name: str, x: torch.Tensor) -> None:
    x_detached = x.detach()
    print(f"\n[{name}]")
    print(f"  shape   : {tuple(x_detached.shape)}")
    print(f"  dtype   : {x_detached.dtype}")
    print(f"  device  : {x_detached.device}")
    print(f"  min/max : {x_detached.min().item():.6f} / {x_detached.max().item():.6f}")
    print(f"  mean    : {x_detached.mean().item():.6f}")
    print(f"  has_nan : {torch.isnan(x_detached).any().item()}")
    print(f"  has_inf : {torch.isinf(x_detached).any().item()}")


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 4
    Nv = 196   # 14x14 image patches
    D = 256
    K = 3      # number of audio/source references
    L_audio = 16

    vision_tokens = torch.randn(B, Nv, D, device=device)
    depth_tokens = torch.randn(B, Nv, D, device=device)
    audio_refs = torch.randn(B, K, L_audio, D, device=device)

    sr_scores, sr_map, stats = build_ori_reference_maps(
        vision_tokens=vision_tokens,
        audio_refs=audio_refs,
        background_refs=None,
        depth_tokens=depth_tokens,
        depth_weight=0.2,
        temperature=1.0,
        grid_size=(14, 14),
    )

    print("\n========== ORI REFERENCE MAP DEBUG: AUDIO-REFERENCE DEPTH-AWARE Sr ==========")
    print(f"device: {device}")
    print_tensor_stats("ref_features", stats["ref_features"])
    print_tensor_stats("fused_tokens", stats["fused_tokens"])
    print_tensor_stats("sr_scores", sr_scores)
    print_tensor_stats("sr_map", sr_map)

    R = K
    assert sr_scores.shape == (B, R, Nv)
    assert sr_map.shape == (B, R, 14, 14)

    for name, tensor in {
        "sr_scores": sr_scores,
        "sr_map": sr_map,
    }.items():
        assert not torch.isnan(tensor).any(), f"{name} has NaN"
        assert not torch.isinf(tensor).any(), f"{name} has Inf"

    print("\nAll depth-aware ORI Sr map checks passed.")


if __name__ == "__main__":
    main()
