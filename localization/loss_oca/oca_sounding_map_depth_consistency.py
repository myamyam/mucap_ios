# Build OCA sounding maps only.
# Kept stages:
# 1) compute_audio_visual_scores
# 2) compute_depth_prior
# 3) build_soft_masks

import math

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
    Compute depth prior per visual patch from depth-token consistency.

    This variant does not use visual-depth token affinity. Instead, each
    depth patch receives a score based on how consistent its depth embedding is
    with neighboring depth patches. For square patch grids, consistency is the
    average cosine similarity to 4-neighbors. For non-square token counts, it
    falls back to average similarity against the other depth tokens in the
    sample.

    Args:
        vision_tokens: [B, Nv, D]
        depth_tokens:  [B, Nd, D]
        tau_d: kept for API compatibility; not used
        mode: kept for API compatibility; not used

    Returns:
        s_d: [B, Nv]
    """
    if vision_tokens.ndim != 3 or depth_tokens.ndim != 3:
        raise ValueError("vision_tokens and depth_tokens must be 3D tensors [B, N, D].")

    if vision_tokens.shape[0] != depth_tokens.shape[0]:
        raise ValueError("Batch size mismatch between vision_tokens and depth_tokens.")

    if vision_tokens.shape[-1] != depth_tokens.shape[-1]:
        raise ValueError("Embedding dim mismatch between vision_tokens and depth_tokens.")

    if vision_tokens.shape[1] != depth_tokens.shape[1]:
        raise ValueError(
            "Depth consistency prior requires vision_tokens and depth_tokens "
            "to have the same patch count."
        )

    d = F.normalize(depth_tokens, dim=-1)    # [B, N, D]
    B, N, D = d.shape

    grid_size = int(math.sqrt(N))

    if grid_size * grid_size == N:
        H = W = grid_size
        d_grid = d.view(B, H, W, D)

        score_sum = d.new_zeros(B, H, W)
        score_count = d.new_zeros(B, H, W)

        if H > 1:
            vertical_sim = (d_grid[:, :-1, :, :] * d_grid[:, 1:, :, :]).sum(dim=-1)
            score_sum[:, :-1, :] += vertical_sim
            score_sum[:, 1:, :] += vertical_sim
            score_count[:, :-1, :] += 1.0
            score_count[:, 1:, :] += 1.0

        if W > 1:
            horizontal_sim = (d_grid[:, :, :-1, :] * d_grid[:, :, 1:, :]).sum(dim=-1)
            score_sum[:, :, :-1] += horizontal_sim
            score_sum[:, :, 1:] += horizontal_sim
            score_count[:, :, :-1] += 1.0
            score_count[:, :, 1:] += 1.0

        consistency = score_sum / score_count.clamp_min(1.0)
        s_d = consistency.view(B, N)
    else:
        sim_dd = torch.matmul(d, d.transpose(1, 2))  # [B, N, N]
        eye = torch.eye(N, device=d.device, dtype=torch.bool).unsqueeze(0)
        sim_dd = sim_dd.masked_fill(eye, 0.0)
        s_d = sim_dd.sum(dim=-1) / max(N - 1, 1)

    return ((s_d + 1.0) * 0.5).clamp(0.0, 1.0)


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
