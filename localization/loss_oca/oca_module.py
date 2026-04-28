#oca 전체 파이프라인을 묶는 상위 모듈
#class OCAloss, forward 등 두고 내부에서 함수 호출하도록
#입력 받아서 최종 loss 내보내는 orchestration 담당 



import torch
import torch.nn as nn

#%%
from oca_ops import (
    compute_audio_visual_scores,
    compute_depth_prior,
    build_soft_masks,
    masked_token_pool,
    compute_oca_loss,
)

#%%

class OCALoss(nn.Module):
    """
    OCA loss module.

    Expected inputs:
        vision_tokens:  [B, Nv, D]
        audio_tokens:   [B, Na, D]
        depth_tokens:   [B, Nd, D]
        fg_text_tokens: [B, K, L, D] or [B, L, D]
        bg_text_tokens: [B, L, D]

    Returns:
        loss_oca: scalar tensor
        stats: dict of intermediate tensors / losses for debugging
    """

    def __init__(
        self,
        tau_a: float = 0.07,
        audio_reduction: str = "attn",
        tau_d: float = 0.07,
        depth_mode: str = "aligned",
        lambda_a: float = 1.0,
        lambda_d: float = 0.2,
        alpha_p: float = 0.4,
        alpha_n: float = 0.4,
        omega: float = 0.1,
        fuse_depth: bool = True,
        depth_weight: float = 0.2,
        tau_neg: float = 0.3,
        temperature: float = 1.0,
        use_soft_negative: bool = True,
    ):
        super().__init__()

        self.tau_a = tau_a
        self.audio_reduction = audio_reduction

        self.tau_d = tau_d
        self.depth_mode = depth_mode

        self.lambda_a = lambda_a
        self.lambda_d = lambda_d
        self.alpha_p = alpha_p
        self.alpha_n = alpha_n
        self.omega = omega

        self.fuse_depth = fuse_depth
        self.depth_weight = depth_weight

        self.tau_neg = tau_neg
        self.temperature = temperature
        self.use_soft_negative = use_soft_negative

    def forward( #입력 토큰들로부터 score->mask->pool->loss 계산
        self,
        vision_tokens: torch.Tensor,
        audio_tokens: torch.Tensor,
        depth_tokens: torch.Tensor,
        fg_text_tokens: torch.Tensor,
        bg_text_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            vision_tokens:  [B, Nv, D]
            audio_tokens:   [B, Na, D]
            depth_tokens:   [B, Nd, D]
            fg_text_tokens: [B, K, L, D] or [B, L, D]
            bg_text_tokens: [B, L, D]

        Returns:
            loss_oca: scalar
            stats: dictionary with intermediate outputs
        """

        # 1) Audio -> visual patch-level scores
        s_a = compute_audio_visual_scores(
            vision_tokens=vision_tokens,
            audio_tokens=audio_tokens,
            tau_a=self.tau_a,
            reduction=self.audio_reduction,
        )  # [B, Nv]

        # 2) Depth prior per visual patch
        s_d = compute_depth_prior(
            vision_tokens=vision_tokens,
            depth_tokens=depth_tokens,
            tau_d=self.tau_d,
            mode=self.depth_mode,
        )  # [B, Nv]

        # 3) Build soft foreground/background masks
        fused_scores, m_fg, m_bg = build_soft_masks(
            audio_scores=s_a,
            depth_scores=s_d,
            lambda_a=self.lambda_a,
            lambda_d=self.lambda_d,
            alpha_p=self.alpha_p,
            alpha_n=self.alpha_n,
            omega=self.omega,
        )  # [B, Nv], [B, Nv], [B, Nv]

        # 4) Pool vision tokens into fg/bg summaries
        z_v_fg, z_v_bg = masked_token_pool(
            tokens=vision_tokens,
            fg_mask=m_fg,
            bg_mask=m_bg,
        )  # [B, D], [B, D]

        # 5) Pool depth tokens into fg/bg summaries
        z_d_fg, z_d_bg = masked_token_pool(
            tokens=depth_tokens,
            fg_mask=m_fg,
            bg_mask=m_bg,
        )  # [B, D], [B, D]

        # 6~9) Text anchor + foreground/background loss + final OCA loss
        loss_oca, oca_stats = compute_oca_loss(
            vision_fg=z_v_fg,
            vision_bg=z_v_bg,
            depth_fg=z_d_fg,
            depth_bg=z_d_bg,
            fg_text_tokens=fg_text_tokens,
            bg_text_tokens=bg_text_tokens,
            fuse_depth=self.fuse_depth,
            depth_weight=self.depth_weight,
            tau_neg=self.tau_neg,
            temperature=self.temperature,
            use_soft_negative=self.use_soft_negative,
        )

        stats = {
            "s_a": s_a.detach(),
            "s_d": s_d.detach(),
            "fused_scores": fused_scores.detach(),
            "m_fg": m_fg.detach(),
            "m_bg": m_bg.detach(),
            "z_v_fg": z_v_fg.detach(),
            "z_v_bg": z_v_bg.detach(),
            "z_d_fg": z_d_fg.detach(),
            "z_d_bg": z_d_bg.detach(),
        }
        stats.update(oca_stats)

        return loss_oca, stats