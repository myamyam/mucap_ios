#잘실행되는지 알아보기 위한 
# #loss_oca 가 숫자로 나오면 1차 통과
#nan, inf면 문제있는것

#shape, mask 범위 맞는지 알아보기 


import torch

from oca_module import OCALoss

#%%

def print_tensor_stats(name: str, x: torch.Tensor) -> None:
    x_detached = x.detach()
    print(f"\n[{name}]")
    print(f"  shape   : {tuple(x_detached.shape)}")
    print(f"  dtype   : {x_detached.dtype}")
    print(f"  min/max : {x_detached.min().item():.6f} / {x_detached.max().item():.6f}")
    print(f"  mean    : {x_detached.mean().item():.6f}")
    print(f"  has_nan : {torch.isnan(x_detached).any().item()}")
    print(f"  has_inf : {torch.isinf(x_detached).any().item()}")

#%%

def main():
    torch.manual_seed(42)

    # -----------------------------
    # Dummy input shapes
    # -----------------------------
    B = 4
    Nv = 196   # image patches, e.g. 14x14
    Na = 128   # audio spectrogram patches/tokens
    Nd = 196   # depth patches (aligned with image patches)
    D = 256    # embedding dim
    K = 2      # number of foreground captions
    L_fg = 16  # foreground text token length
    L_bg = 12  # background text token length

    # -----------------------------
    # Dummy tensors
    # -----------------------------
    vision_tokens = torch.randn(B, Nv, D)
    audio_tokens = torch.randn(B, Na, D)
    depth_tokens = torch.randn(B, Nd, D)

    # foreground text: [B, K, L_fg, D]
    fg_text_tokens = torch.randn(B, K, L_fg, D)

    # background text: [B, L_bg, D]
    bg_text_tokens = torch.randn(B, L_bg, D)

    # -----------------------------
    # OCA module
    # -----------------------------
    oca_loss_fn = OCALoss(
        tau_a=0.07,
        audio_reduction="attn",
        tau_d=0.07,
        depth_mode="aligned",
        lambda_a=1.0,
        lambda_d=0.2,
        alpha_p=0.4,
        alpha_n=0.4,
        omega=0.1,
        fuse_depth=True,
        depth_weight=0.2,
        tau_neg=0.3,
        temperature=1.0,
        use_soft_negative=True,
    )

    # -----------------------------
    # Forward
    # -----------------------------
    loss_oca, stats = oca_loss_fn(
        vision_tokens=vision_tokens,
        audio_tokens=audio_tokens,
        depth_tokens=depth_tokens,
        fg_text_tokens=fg_text_tokens,
        bg_text_tokens=bg_text_tokens,
    )

    # -----------------------------
    # Basic checks
    # -----------------------------
    print("\n========== OCA DEBUG ==========")
    print(f"loss_oca   : {loss_oca.item():.6f}")
    print(f"loss_frg   : {stats['loss_frg'].item():.6f}")
    print(f"loss_bkg   : {stats['loss_bkg'].item():.6f}")
    print(f"loss is nan: {torch.isnan(loss_oca).item()}")
    print(f"loss is inf: {torch.isinf(loss_oca).item()}")

    # -----------------------------
    # Shape / range checks
    # -----------------------------
    print_tensor_stats("s_a", stats["s_a"])                   # [B, Nv]
    print_tensor_stats("s_d", stats["s_d"])                   # [B, Nv]
    print_tensor_stats("fused_scores", stats["fused_scores"]) # [B, Nv]

    print_tensor_stats("m_fg", stats["m_fg"])                 # [B, Nv]
    print_tensor_stats("m_bg", stats["m_bg"])                 # [B, Nv]

    print_tensor_stats("z_v_fg", stats["z_v_fg"])             # [B, D]
    print_tensor_stats("z_v_bg", stats["z_v_bg"])             # [B, D]
    print_tensor_stats("z_d_fg", stats["z_d_fg"])             # [B, D]
    print_tensor_stats("z_d_bg", stats["z_d_bg"])             # [B, D]

    print_tensor_stats("fg_anchor", stats["fg_anchor"])       # [B, D]
    print_tensor_stats("bg_anchor", stats["bg_anchor"])       # [B, D]
    print_tensor_stats("fg_region", stats["fg_region"])       # [B, D]
    print_tensor_stats("bg_region", stats["bg_region"])       # [B, D]

    # -----------------------------
    # Sanity assertions
    # -----------------------------
    assert stats["s_a"].shape == (B, Nv)
    assert stats["s_d"].shape == (B, Nv)
    assert stats["fused_scores"].shape == (B, Nv)

    assert stats["m_fg"].shape == (B, Nv)
    assert stats["m_bg"].shape == (B, Nv)

    assert stats["z_v_fg"].shape == (B, D)
    assert stats["z_v_bg"].shape == (B, D)
    assert stats["z_d_fg"].shape == (B, D)
    assert stats["z_d_bg"].shape == (B, D)

    assert stats["fg_anchor"].shape == (B, D)
    assert stats["bg_anchor"].shape == (B, D)
    assert stats["fg_region"].shape == (B, D)
    assert stats["bg_region"].shape == (B, D)

    assert not torch.isnan(loss_oca).any()
    assert not torch.isinf(loss_oca).any()

    # mask range check
    assert stats["m_fg"].min() >= 0.0 and stats["m_fg"].max() <= 1.0
    assert stats["m_bg"].min() >= 0.0 and stats["m_bg"].max() <= 1.0

    print("\nAll debug checks passed.")


if __name__ == "__main__":
    main()