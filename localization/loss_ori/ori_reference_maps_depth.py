import math

import torch
import torch.nn.functional as F


def pool_audio_references(audio_refs: torch.Tensor) -> torch.Tensor:
    """
    Convert audio/source reference tokens into one feature per sound source.

    Args:
        audio_refs:
            [B, K, L, D] audio/source tokens per source, or
            [B, K, D] already pooled audio-source references.

    Returns:
        audio_features: [B, K, D]
    """
    if audio_refs.ndim == 4:
        return audio_refs.mean(dim=2)
    if audio_refs.ndim == 3:
        return audio_refs
    raise ValueError("audio_refs must be [B, K, L, D] or [B, K, D].")


def pool_audio_tokens_as_single_reference(audio_tokens: torch.Tensor) -> torch.Tensor:
    """
    Pool one audio token sequence into a single ORI source reference.

    Args:
        audio_tokens: [B, Na, D] audio patch/token sequence.

    Returns:
        audio_refs: [B, 1, D]
    """
    if audio_tokens.ndim != 3:
        raise ValueError("audio_tokens must be [B, Na, D].")
    return audio_tokens.mean(dim=1, keepdim=True)


def pool_background_reference(background_refs: torch.Tensor) -> torch.Tensor:
    """
    Convert an optional background reference into one background feature.

    Args:
        background_refs:
            [B, L, D] background reference tokens, or
            [B, D] already pooled background reference.

    Returns:
        bg_feature: [B, D]
    """
    if background_refs.ndim == 3:
        return background_refs.mean(dim=1)
    if background_refs.ndim == 2:
        return background_refs
    raise ValueError("background_refs must be [B, L, D] or [B, D].")


def build_reference_features(
    audio_refs: torch.Tensor,
    background_refs: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Build ORI audio/source-reference features.

    Args:
        audio_refs: [B, K, L, D] or [B, K, D]
        background_refs: [B, L, D], [B, D], or None

    Returns:
        ref_features: [B, K, D] for audio/source references.
            If background_refs is provided, returns [B, K + 1, D]
            with the optional background reference last.
    """
    audio_features = pool_audio_references(audio_refs)

    if background_refs is None:
        return audio_features

    bg_feature = pool_background_reference(background_refs)

    if audio_features.shape[0] != bg_feature.shape[0]:
        raise ValueError("Batch size mismatch between audio_refs and background_refs.")
    if audio_features.shape[-1] != bg_feature.shape[-1]:
        raise ValueError("Embedding dim mismatch between audio_refs and background_refs.")

    return torch.cat([audio_features, bg_feature.unsqueeze(1)], dim=1)


def fuse_vision_depth_tokens(
    vision_tokens: torch.Tensor,
    depth_tokens: torch.Tensor | None = None,
    depth_weight: float = 0.2,
) -> torch.Tensor:
    """
    Build depth-aware visual tokens for the ORI reference map branch.

    Args:
        vision_tokens: [B, Nv, D]
        depth_tokens: [B, Nv, D] or None
        depth_weight: depth contribution when depth_tokens is provided

    Returns:
        fused_tokens: [B, Nv, D]
    """
    if vision_tokens.ndim != 3:
        raise ValueError("vision_tokens must be [B, Nv, D].")

    if depth_tokens is None:
        return vision_tokens

    if depth_tokens.ndim != 3:
        raise ValueError("depth_tokens must be [B, Nv, D].")
    if depth_tokens.shape != vision_tokens.shape:
        raise ValueError("depth_tokens must have the same shape as vision_tokens.")

    return vision_tokens + depth_weight * depth_tokens


def compute_reference_associated_scores(
    vision_tokens: torch.Tensor,
    ref_features: torch.Tensor,
    depth_tokens: torch.Tensor | None = None,
    depth_weight: float = 0.2,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute ORI reference-associated patch scores Sr.

    Args:
        vision_tokens: [B, Nv, D]
        ref_features: [B, R, D], where R is K or K + 1
        depth_tokens: [B, Nv, D] or None
        depth_weight: depth contribution for depth-aware visual tokens
        temperature: score temperature

    Returns:
        sr_scores: [B, R, Nv]
    """
    if ref_features.ndim != 3:
        raise ValueError("ref_features must be [B, R, D].")
    if vision_tokens.shape[0] != ref_features.shape[0]:
        raise ValueError("Batch size mismatch between vision_tokens and ref_features.")
    if vision_tokens.shape[-1] != ref_features.shape[-1]:
        raise ValueError("Embedding dim mismatch between vision_tokens and ref_features.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    fused_tokens = fuse_vision_depth_tokens(
        vision_tokens=vision_tokens,
        depth_tokens=depth_tokens,
        depth_weight=depth_weight,
    )

    v = F.normalize(fused_tokens, dim=-1)    # [B, Nv, D]
    r = F.normalize(ref_features, dim=-1)    # [B, R, D]

    sr_scores = torch.einsum("bnd,brd->brn", v, r) / temperature
    return sr_scores


def infer_square_grid_size(num_patches: int) -> tuple[int, int]:
    grid_size = int(math.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        raise ValueError(
            "Cannot infer square grid size from num_patches. "
            "Pass grid_size=(H, W) explicitly."
        )
    return grid_size, grid_size


def reshape_sr_to_map(
    sr_scores: torch.Tensor,
    grid_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """
    Reshape Sr from patch sequence to spatial map.

    Args:
        sr_scores: [B, R, Nv]
        grid_size: (H, W) or None. If None, Nv must be a square number.

    Returns:
        sr_map: [B, R, H, W]
    """
    if sr_scores.ndim != 3:
        raise ValueError("sr_scores must be [B, R, Nv].")

    B, R, Nv = sr_scores.shape
    H, W = infer_square_grid_size(Nv) if grid_size is None else grid_size

    if H * W != Nv:
        raise ValueError("grid_size must satisfy H * W == Nv.")

    return sr_scores.view(B, R, H, W)


def build_ori_reference_maps(
    vision_tokens: torch.Tensor,
    audio_refs: torch.Tensor,
    background_refs: torch.Tensor | None = None,
    depth_tokens: torch.Tensor | None = None,
    depth_weight: float = 0.2,
    temperature: float = 1.0,
    grid_size: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Build ORI-style audio-reference-associated maps up to Sr.

    This is the parallel ORI branch map extractor. It does not compute ORI loss.

    Args:
        vision_tokens: [B, Nv, D]
        audio_refs: [B, K, L, D] or [B, K, D]
            Source-level audio references. If only one full audio-token
            sequence is available as [B, Na, D], call
            pool_audio_tokens_as_single_reference(audio_tokens) first.
        background_refs: [B, L, D], [B, D], or None
        depth_tokens: [B, Nv, D] or None
        depth_weight: depth contribution for depth-aware visual tokens
        temperature: cosine score temperature
        grid_size: (H, W) or None. If None, Nv must be square.

    Returns:
        sr_scores: [B, R, Nv]
        sr_map: [B, R, H, W]
        stats: detached intermediates for debugging

    Notes:
        Sr = similarity(vision_depth_patch, audio_reference).
        R = K when background_refs is None.
        R = K + 1 when background_refs is provided, with the optional
        background map last.
    """
    ref_features = build_reference_features(
        audio_refs=audio_refs,
        background_refs=background_refs,
    )
    fused_tokens = fuse_vision_depth_tokens(
        vision_tokens=vision_tokens,
        depth_tokens=depth_tokens,
        depth_weight=depth_weight,
    )

    sr_scores = compute_reference_associated_scores(
        vision_tokens=vision_tokens,
        ref_features=ref_features,
        depth_tokens=depth_tokens,
        depth_weight=depth_weight,
        temperature=temperature,
    )
    sr_map = reshape_sr_to_map(sr_scores=sr_scores, grid_size=grid_size)

    stats = {
        "ref_features": ref_features.detach(),
        "fused_tokens": fused_tokens.detach(),
        "sr_scores": sr_scores.detach(),
        "sr_map": sr_map.detach(),
    }

    return sr_scores, sr_map, stats
