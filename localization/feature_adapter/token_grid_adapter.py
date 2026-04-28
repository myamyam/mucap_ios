import math

import torch
import torch.nn.functional as F


def infer_square_grid_size(num_tokens: int) -> tuple[int, int]:
    """
    Infer a square spatial grid from a token count.

    Args:
        num_tokens: number of spatial tokens, e.g. 196 for 14x14.

    Returns:
        (H, W)
    """
    grid_size = int(math.sqrt(num_tokens))
    if grid_size * grid_size != num_tokens:
        raise ValueError(
            "Cannot infer a square grid from num_tokens. "
            "Pass grid_size=(H, W) explicitly."
        )
    return grid_size, grid_size


def tokens_to_spatial_grid(
    tokens: torch.Tensor,
    grid_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """
    Convert token sequence layout to spatial feature-map layout.

    Args:
        tokens: [B, N, D]
        grid_size: (H, W). If None, N must be a square number.

    Returns:
        grid: [B, D, H, W]
    """
    if tokens.ndim != 3:
        raise ValueError("tokens must be [B, N, D].")

    B, N, D = tokens.shape
    H, W = infer_square_grid_size(N) if grid_size is None else grid_size

    if H * W != N:
        raise ValueError("grid_size must satisfy H * W == N.")

    return tokens.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()


def spatial_grid_to_tokens(grid: torch.Tensor) -> torch.Tensor:
    """
    Convert spatial feature-map layout back to token sequence layout.

    Args:
        grid: [B, D, H, W]

    Returns:
        tokens: [B, H * W, D]
    """
    if grid.ndim != 4:
        raise ValueError("grid must be [B, D, H, W].")

    B, D, H, W = grid.shape
    return grid.permute(0, 2, 3, 1).contiguous().view(B, H * W, D)


def resize_token_grid(
    tokens: torch.Tensor,
    target_grid_size: tuple[int, int],
    source_grid_size: tuple[int, int] | None = None,
    mode: str = "bilinear",
    align_corners: bool = False,
) -> torch.Tensor:
    """
    Resize spatial tokens from one patch grid to another.

    Args:
        tokens: [B, N, D]
        target_grid_size: target (H, W)
        source_grid_size: source (H, W). If None, N must be square.
        mode: interpolation mode passed to torch.nn.functional.interpolate
        align_corners: interpolation align_corners option for linear modes

    Returns:
        resized_tokens: [B, target_H * target_W, D]
    """
    grid = tokens_to_spatial_grid(tokens, grid_size=source_grid_size)

    if grid.shape[-2:] == target_grid_size:
        return spatial_grid_to_tokens(grid)

    interpolate_kwargs = {
        "input": grid,
        "size": target_grid_size,
        "mode": mode,
    }
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        interpolate_kwargs["align_corners"] = align_corners

    resized_grid = F.interpolate(**interpolate_kwargs)
    return spatial_grid_to_tokens(resized_grid)


def align_depth_tokens_to_vision(
    vision_tokens: torch.Tensor,
    depth_tokens: torch.Tensor,
    vision_grid_size: tuple[int, int] | None = None,
    depth_grid_size: tuple[int, int] | None = None,
    mode: str = "bilinear",
    align_corners: bool = False,
) -> torch.Tensor:
    """
    Resize depth patch tokens so they match the vision token grid.

    Args:
        vision_tokens: [B, Nv, D]
        depth_tokens: [B, Nd, D]
        vision_grid_size: vision token grid (H, W). If None, Nv must be square.
        depth_grid_size: depth token grid (H, W). If None, Nd must be square.
        mode: interpolation mode for depth-token resizing
        align_corners: interpolation align_corners option for linear modes

    Returns:
        aligned_depth_tokens: [B, Nv, D]
    """
    if vision_tokens.ndim != 3 or depth_tokens.ndim != 3:
        raise ValueError("vision_tokens and depth_tokens must both be [B, N, D].")
    if vision_tokens.shape[0] != depth_tokens.shape[0]:
        raise ValueError("Batch size mismatch between vision_tokens and depth_tokens.")
    if vision_tokens.shape[-1] != depth_tokens.shape[-1]:
        raise ValueError("Embedding dim mismatch between vision_tokens and depth_tokens.")

    _, Nv, _ = vision_tokens.shape
    _, Nd, _ = depth_tokens.shape
    target_grid_size = (
        infer_square_grid_size(Nv) if vision_grid_size is None else vision_grid_size
    )

    if target_grid_size[0] * target_grid_size[1] != Nv:
        raise ValueError("vision_grid_size must satisfy H * W == vision token count.")
    if depth_grid_size is not None and depth_grid_size[0] * depth_grid_size[1] != Nd:
        raise ValueError("depth_grid_size must satisfy H * W == depth token count.")

    if Nv == Nd and (depth_grid_size is None or depth_grid_size == target_grid_size):
        return depth_tokens

    return resize_token_grid(
        tokens=depth_tokens,
        target_grid_size=target_grid_size,
        source_grid_size=depth_grid_size,
        mode=mode,
        align_corners=align_corners,
    )
