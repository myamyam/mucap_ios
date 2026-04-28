import torch

from oca_sounding_map_depth_consistency import (
    build_soft_masks,
    compute_audio_visual_scores,
    compute_depth_prior,
)


def print_tensor_stats(name: str, x: torch.Tensor) -> None:
    x_detached = x.detach()
    print(f"\n[{name}]")
    print(f"  shape   : {tuple(x_detached.shape)}")
    print(f"  dtype   : {x_detached.dtype}")
    print(f"  min/max : {x_detached.min().item():.6f} / {x_detached.max().item():.6f}")
    print(f"  mean    : {x_detached.mean().item():.6f}")
    print(f"  has_nan : {torch.isnan(x_detached).any().item()}")
    print(f"  has_inf : {torch.isinf(x_detached).any().item()}")


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 4
    Nv = 196   # image patches, e.g. 14x14
    Na = 128   # audio spectrogram patches/tokens
    Nd = 196   # depth consistency requires Nd == Nv
    D = 256    # embedding dim

    vision_tokens = torch.randn(B, Nv, D, device=device)
    audio_tokens = torch.randn(B, Na, D, device=device)
    depth_tokens = torch.randn(B, Nd, D, device=device)

    s_a = compute_audio_visual_scores(
        vision_tokens=vision_tokens,
        audio_tokens=audio_tokens,
        tau_a=0.07,
        reduction="attn",
    )

    s_d = compute_depth_prior(
        vision_tokens=vision_tokens,
        depth_tokens=depth_tokens,
        tau_d=0.07,
        mode="aligned",
    )

    fused_scores, m_fg, m_bg = build_soft_masks(
        audio_scores=s_a,
        depth_scores=s_d,
        lambda_a=1.0,
        lambda_d=0.2,
        alpha_p=0.4,
        alpha_n=0.4,
        omega=0.1,
    )

    print("\n========== OCA SOUNDING MAP DEBUG: DEPTH CONSISTENCY ==========")
    print(f"device: {device}")
    print(f"vision_tokens.device: {vision_tokens.device}")
    print(f"audio_tokens.device : {audio_tokens.device}")
    print(f"depth_tokens.device : {depth_tokens.device}")
    print(f"m_fg.device         : {m_fg.device}")
    print_tensor_stats("s_a", s_a)
    print_tensor_stats("s_d", s_d)
    print_tensor_stats("fused_scores", fused_scores)
    print_tensor_stats("m_fg", m_fg)
    print_tensor_stats("m_bg", m_bg)

    assert s_a.shape == (B, Nv)
    assert s_d.shape == (B, Nv)
    assert fused_scores.shape == (B, Nv)
    assert m_fg.shape == (B, Nv)
    assert m_bg.shape == (B, Nv)

    for name, tensor in {
        "s_a": s_a,
        "s_d": s_d,
        "fused_scores": fused_scores,
        "m_fg": m_fg,
        "m_bg": m_bg,
    }.items():
        assert not torch.isnan(tensor).any(), f"{name} has NaN"
        assert not torch.isinf(tensor).any(), f"{name} has Inf"

    assert s_d.min() >= 0.0 and s_d.max() <= 1.0
    assert m_fg.min() >= 0.0 and m_fg.max() <= 1.0
    assert m_bg.min() >= 0.0 and m_bg.max() <= 1.0

    m_fg_map = m_fg.view(B, 14, 14)
    m_bg_map = m_bg.view(B, 14, 14)
    print_tensor_stats("m_fg_map", m_fg_map)
    print_tensor_stats("m_bg_map", m_bg_map)

    print("\nAll depth-consistency sounding map checks passed.")


if __name__ == "__main__":
    main()
