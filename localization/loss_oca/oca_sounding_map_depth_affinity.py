# Build OCA sounding maps only.
# Kept stages:
# 1) compute_audio_visual_scores
# 2) compute_depth_prior
# 3) build_soft_masks

import torch
import torch.nn.functional as F


def compute_audio_visual_scores(
    vision_tokens: torch.Tensor,
    audio_tokens: torch.Tensor,
    tau_a: float = 0.07,
    reduction: str = "attn",
) -> torch.Tensor:
    """
    Compute patch-level sounding scores from vision/audio tokens.

    Args:
        vision_tokens: [B, Nv, D]
        audio_tokens:  [B, Na, D]
        tau_a: temperature for attention-style reduction
        reduction: one of {"attn", "mean", "max"}

    Returns:
        s_a: [B, Nv]
    """
    if vision_tokens.ndim != 3 or audio_tokens.ndim != 3:
        raise ValueError("vision_tokens and audio_tokens must be 3D tensors [B, N, D].")

    if vision_tokens.shape[0] != audio_tokens.shape[0]:
        raise ValueError("Batch size mismatch between vision_tokens and audio_tokens.")

    if vision_tokens.shape[-1] != audio_tokens.shape[-1]:
        raise ValueError("Embedding dim mismatch between vision_tokens and audio_tokens.")

    v = F.normalize(vision_tokens, dim=-1)   # [B, Nv, D]
    a = F.normalize(audio_tokens, dim=-1)    # [B, Na, D]

    sim_va = torch.matmul(v, a.transpose(1, 2))  # [B, Nv, Na]

    if reduction == "attn":
        attn = F.softmax(sim_va / tau_a, dim=-1)     # [B, Nv, Na]
        s_a = (attn * sim_va).sum(dim=-1)            # [B, Nv]
    elif reduction == "mean":
        s_a = sim_va.mean(dim=-1)                    # [B, Nv]
    elif reduction == "max":
        s_a = sim_va.max(dim=-1).values              # [B, Nv]
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")

    return s_a


def compute_depth_prior(
    vision_tokens: torch.Tensor,
    depth_tokens: torch.Tensor,
    tau_d: float = 0.07,
    mode: str = "aligned",
) -> torch.Tensor:
    """
    Compute depth prior per visual patch.

    Args:
        vision_tokens: [B, Nv, D]
        depth_tokens:  [B, Nd, D]
        tau_d: temperature for attention-style reduction
        mode: one of {"aligned", "attn"}

    Returns:
        s_d: [B, Nv]
    """
    if vision_tokens.ndim != 3 or depth_tokens.ndim != 3:
        raise ValueError("vision_tokens and depth_tokens must be 3D tensors [B, N, D].")

    if vision_tokens.shape[0] != depth_tokens.shape[0]:
        raise ValueError("Batch size mismatch between vision_tokens and depth_tokens.")

    if vision_tokens.shape[-1] != depth_tokens.shape[-1]:
        raise ValueError("Embedding dim mismatch between vision_tokens and depth_tokens.")

    v = F.normalize(vision_tokens, dim=-1)   # [B, Nv, D]
    d = F.normalize(depth_tokens, dim=-1)    # [B, Nd, D]

    if mode == "aligned":
        if vision_tokens.shape[1] != depth_tokens.shape[1]:
            raise ValueError(
                "For mode='aligned', vision_tokens and depth_tokens must have same token count."
            )
        s_d = (v * d).sum(dim=-1)  # [B, Nv]

    elif mode == "attn":
        sim_vd = torch.matmul(v, d.transpose(1, 2))   # [B, Nv, Nd]
        attn = F.softmax(sim_vd / tau_d, dim=-1)      # [B, Nv, Nd]
        s_d = (attn * sim_vd).sum(dim=-1)             # [B, Nv]

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return s_d


def build_soft_masks(
    audio_scores: torch.Tensor,
    depth_scores: torch.Tensor | None = None,
    lambda_a: float = 1.0,
    lambda_d: float = 0.2,
    alpha_p: float = 0.4,
    alpha_n: float = 0.4,
    omega: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fuse audio/depth scores and build soft foreground/background masks.

    Args:
        audio_scores: [B, Nv]
        depth_scores: [B, Nv] or None
        lambda_a: weight for audio score
        lambda_d: weight for depth prior
        alpha_p: foreground threshold center
        alpha_n: background threshold center
        omega: softness temperature

    Returns:
        fused_scores: [B, Nv]
        m_fg: [B, Nv]
        m_bg: [B, Nv]
    """
    if audio_scores.ndim != 2:
        raise ValueError("audio_scores must be [B, Nv].")

    fused_scores = lambda_a * audio_scores

    if depth_scores is not None:
        if depth_scores.shape != audio_scores.shape:
            raise ValueError("depth_scores must have same shape as audio_scores.")
        fused_scores = fused_scores + lambda_d * depth_scores

    m_fg = torch.sigmoid((fused_scores - alpha_p) / omega)          # [B, Nv]
    m_bg = 1.0 - torch.sigmoid((fused_scores - alpha_n) / omega)    # [B, Nv]

    return fused_scores, m_fg, m_bg
