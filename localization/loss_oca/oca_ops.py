#until before loss calculation
#총 8개 : 1) compute_audio_visual_scores, 2) compute_depth_prior, 3) build_soft_masks, 4) masked_token_pool
        #5) build_text_anchor, 6) foreground_oca_loss, 7) background_oca_loss, 8) OCAloss.forward



#1) compute_audio_visual_scores, 2) compute_depth_prior, 3) build_soft_masks, 4) masked_token_pool

import torch
import torch.nn as nn
import torch.nn.functional as F

#%%

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

    # Normalize for cosine similarity
    v = F.normalize(vision_tokens, dim=-1)   # [B, Nv, D]
    a = F.normalize(audio_tokens, dim=-1)    # [B, Na, D]

    # Pairwise cosine similarity: [B, Nv, Na]
    sim_va = torch.matmul(v, a.transpose(1, 2))

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

#%%

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
        # Patch i in vision <-> patch i in depth
        s_d = (v * d).sum(dim=-1)  # [B, Nv]

    elif mode == "attn":
        sim_vd = torch.matmul(v, d.transpose(1, 2))   # [B, Nv, Nd]
        attn = F.softmax(sim_vd / tau_d, dim=-1)      # [B, Nv, Nd]
        s_d = (attn * sim_vd).sum(dim=-1)             # [B, Nv]

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return s_d

#%%

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

#%%

def masked_token_pool(
    tokens: torch.Tensor,
    fg_mask: torch.Tensor,
    bg_mask: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pool token sequences into foreground/background summaries.

    Args:
        tokens:   [B, N, D]
        fg_mask:  [B, N]
        bg_mask:  [B, N]
        eps: numerical stability

    Returns:
        z_fg: [B, D]
        z_bg: [B, D]
    """
    if tokens.ndim != 3:
        raise ValueError("tokens must be [B, N, D].")
    if fg_mask.ndim != 2 or bg_mask.ndim != 2:
        raise ValueError("fg_mask and bg_mask must be [B, N].")
    if tokens.shape[:2] != fg_mask.shape or tokens.shape[:2] != bg_mask.shape:
        raise ValueError("Mask shapes must match tokens' first two dims [B, N].")

    fg_w = fg_mask.unsqueeze(-1)  # [B, N, 1]
    bg_w = bg_mask.unsqueeze(-1)  # [B, N, 1]

    z_fg = (tokens * fg_w).sum(dim=1) / (fg_w.sum(dim=1) + eps)  # [B, D]
    z_bg = (tokens * bg_w).sum(dim=1) / (bg_w.sum(dim=1) + eps)  # [B, D]

    return z_fg, z_bg






# 5) build_text_anchor, 6) foreground_oca_loss, 7) background_oca_loss, 8) OCAloss.forward

#%%

import torch
import torch.nn.functional as F


def build_text_anchor(
    fg_text_tokens: torch.Tensor,
    bg_text_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build foreground/background text anchors from token-wise text features.

    Args:
        fg_text_tokens:
            Foreground text tokens.
            Shape can be:
                [B, K, L, D]  -> K foreground captions, each with L tokens
            or
                [B, L, D]     -> already merged foreground caption tokens

        bg_text_tokens:
            Background text tokens.
            Shape:
                [B, L, D]

    Returns:
        fg_anchor: [B, D]
        bg_anchor: [B, D]
    """
    if fg_text_tokens.ndim not in (3, 4):
        raise ValueError("fg_text_tokens must be [B, L, D] or [B, K, L, D].")
    if bg_text_tokens.ndim != 3:
        raise ValueError("bg_text_tokens must be [B, L, D].")

    if fg_text_tokens.shape[0] != bg_text_tokens.shape[0]:
        raise ValueError("Batch size mismatch between fg_text_tokens and bg_text_tokens.")

    if fg_text_tokens.shape[-1] != bg_text_tokens.shape[-1]:
        raise ValueError("Embedding dim mismatch between fg_text_tokens and bg_text_tokens.")

    # Foreground anchor
    if fg_text_tokens.ndim == 4:
        # [B, K, L, D] -> average over token dim L, then average over caption dim K
        fg_anchor = fg_text_tokens.mean(dim=2).mean(dim=1)   # [B, D]
    else:
        # [B, L, D] -> average over token dim L
        fg_anchor = fg_text_tokens.mean(dim=1)               # [B, D]

    # Background anchor
    bg_anchor = bg_text_tokens.mean(dim=1)                   # [B, D]

    return fg_anchor, bg_anchor

#%%

def foreground_oca_loss(
    fg_region: torch.Tensor,
    bg_region: torch.Tensor,
    fg_anchor: torch.Tensor,
    tau_neg: float = 0.3,
    temperature: float = 1.0,
    use_soft_negative: bool = True,
) -> torch.Tensor:
    """
    Foreground OCA loss.

    Args:
        fg_region:   [B, D] foreground region summary
        bg_region:   [B, D] background region summary
        fg_anchor:   [B, D] foreground text anchor
        tau_neg:     threshold for filtering soft negatives
        temperature: contrastive temperature
        use_soft_negative: whether to use batch soft negatives

    Returns:
        loss_frg: scalar
    """
    if fg_region.ndim != 2 or bg_region.ndim != 2 or fg_anchor.ndim != 2:
        raise ValueError("fg_region, bg_region, fg_anchor must all be [B, D].")

    if fg_region.shape != bg_region.shape or fg_region.shape != fg_anchor.shape:
        raise ValueError("fg_region, bg_region, fg_anchor must have same shape [B, D].")

    # Normalize
    z_fg = F.normalize(fg_region, dim=-1)      # [B, D]
    z_bg = F.normalize(bg_region, dim=-1)      # [B, D]
    r_fg = F.normalize(fg_anchor, dim=-1)      # [B, D]

    # Positive: fg_region <-> fg_anchor (same sample)
    pos_logits = (z_fg * r_fg).sum(dim=-1) / temperature      # [B]
    pos = torch.exp(pos_logits)                               # [B]

    # Hard negative: bg_region <-> fg_anchor (same sample)
    hard_neg_logits = (z_bg * r_fg).sum(dim=-1) / temperature # [B]
    hard_neg = torch.exp(hard_neg_logits)                     # [B]

    # Soft negatives: other samples' fg_region against current sample's fg_anchor
    if use_soft_negative and z_fg.shape[0] > 1:
        # [B, B] similarities between all fg anchors
        anchor_sim = torch.matmul(r_fg, r_fg.transpose(0, 1))  # [B, B]

        # [B, B] similarities between all fg regions and all fg anchors
        cross_sim = torch.matmul(z_fg, r_fg.transpose(0, 1)) / temperature  # [B, B]

        B = z_fg.shape[0]
        eye = torch.eye(B, device=z_fg.device, dtype=torch.bool)

        # valid negative if j != i and anchor similarity <= tau_neg
        valid_neg_mask = (~eye) & (anchor_sim <= tau_neg)     # [B, B]

        soft_neg = torch.exp(cross_sim) * valid_neg_mask.float()  # [B, B]
        soft_neg = soft_neg.sum(dim=0)                        # [B]
    else:
        soft_neg = torch.zeros_like(pos)

    loss = -torch.log(pos / (pos + hard_neg + soft_neg + 1e-8))
    return loss.mean()

#%%

def background_oca_loss(
    fg_region: torch.Tensor,
    bg_region: torch.Tensor,
    bg_anchor: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Background OCA loss.

    Args:
        fg_region:   [B, D] foreground region summary
        bg_region:   [B, D] background region summary
        bg_anchor:   [B, D] background text anchor
        temperature: contrastive temperature

    Returns:
        loss_bkg: scalar
    """
    if fg_region.ndim != 2 or bg_region.ndim != 2 or bg_anchor.ndim != 2:
        raise ValueError("fg_region, bg_region, bg_anchor must all be [B, D].")

    if fg_region.shape != bg_region.shape or fg_region.shape != bg_anchor.shape:
        raise ValueError("fg_region, bg_region, bg_anchor must have same shape [B, D].")

    # Normalize
    z_fg = F.normalize(fg_region, dim=-1)      # [B, D]
    z_bg = F.normalize(bg_region, dim=-1)      # [B, D]
    r_bg = F.normalize(bg_anchor, dim=-1)      # [B, D]

    # Positive: bg_region <-> bg_anchor
    pos_logits = (z_bg * r_bg).sum(dim=-1) / temperature     # [B]
    pos = torch.exp(pos_logits)                               # [B]

    # Hard negative: fg_region <-> bg_anchor
    hard_neg_logits = (z_fg * r_bg).sum(dim=-1) / temperature # [B]
    hard_neg = torch.exp(hard_neg_logits)                     # [B]

    loss = -torch.log(pos / (pos + hard_neg + 1e-8))
    return loss.mean()

#%%

def compute_oca_loss(
    vision_fg: torch.Tensor,
    vision_bg: torch.Tensor,
    depth_fg: torch.Tensor,
    depth_bg: torch.Tensor,
    fg_text_tokens: torch.Tensor,
    bg_text_tokens: torch.Tensor,
    fuse_depth: bool = True,
    depth_weight: float = 0.2,
    tau_neg: float = 0.3,
    temperature: float = 1.0,
    use_soft_negative: bool = True,
) -> tuple[torch.Tensor, dict]:
    """
    Compute final OCA loss from pooled region summaries + text tokens.

    Args:
        vision_fg:      [B, D]
        vision_bg:      [B, D]
        depth_fg:       [B, D]
        depth_bg:       [B, D]
        fg_text_tokens: [B, K, L, D] or [B, L, D]
        bg_text_tokens: [B, L, D]
        fuse_depth:     whether to fuse depth summaries into region summaries
        depth_weight:   weight for depth summary fusion
        tau_neg:        threshold for soft negative filtering
        temperature:    contrastive temperature
        use_soft_negative: whether to use batch soft negatives in foreground loss

    Returns:
        loss_oca: scalar
        stats: dict with:
            - loss_frg
            - loss_bkg
            - fg_anchor
            - bg_anchor
            - fg_region
            - bg_region
    """
    if vision_fg.ndim != 2 or vision_bg.ndim != 2 or depth_fg.ndim != 2 or depth_bg.ndim != 2:
        raise ValueError("vision_fg, vision_bg, depth_fg, depth_bg must all be [B, D].")

    if not (vision_fg.shape == vision_bg.shape == depth_fg.shape == depth_bg.shape):
        raise ValueError("vision/depth foreground/background summaries must all have same shape [B, D].")

    # 5) Build text anchors
    fg_anchor, bg_anchor = build_text_anchor(
        fg_text_tokens=fg_text_tokens,
        bg_text_tokens=bg_text_tokens,
    )  # both [B, D]

    # region fusion
    if fuse_depth:
        fg_region = F.normalize(vision_fg + depth_weight * depth_fg, dim=-1)  # [B, D]
        bg_region = F.normalize(vision_bg + depth_weight * depth_bg, dim=-1)  # [B, D]
    else:
        fg_region = F.normalize(vision_fg, dim=-1)
        bg_region = F.normalize(vision_bg, dim=-1)

    # 6) Foreground loss
    loss_frg = foreground_oca_loss(
        fg_region=fg_region,
        bg_region=bg_region,
        fg_anchor=fg_anchor,
        tau_neg=tau_neg,
        temperature=temperature,
        use_soft_negative=use_soft_negative,
    )

    # 7) Background loss
    loss_bkg = background_oca_loss(
        fg_region=fg_region,
        bg_region=bg_region,
        bg_anchor=bg_anchor,
        temperature=temperature,
    )

    # 8) Final OCA loss
    loss_oca = 0.5 * (loss_frg + loss_bkg)

    stats = {
        "loss_frg": loss_frg.detach(),
        "loss_bkg": loss_bkg.detach(),
        "fg_anchor": fg_anchor.detach(),
        "bg_anchor": bg_anchor.detach(),
        "fg_region": fg_region.detach(),
        "bg_region": bg_region.detach(),
    }

    return loss_oca, stats

