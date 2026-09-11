"""
Adaptive Instance Normalization (AdaIN) module.
Aligns channel-wise mean and variance of content representations to match style statistics.
Supports continuous latent feature interpolation for controllable style strength.
"""

import torch
import torch.nn as nn
from typing import Tuple


def calc_mean_std(feat: torch.Tensor, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes channel-wise spatial mean and standard deviation for each sample in a batch.
    Args:
        feat: Tensor of shape (B, C, H, W)
        eps: Small constant to avoid division by zero
    Returns:
        mean: Tensor of shape (B, C, 1, 1)
        std: Tensor of shape (B, C, 1, 1)
    """
    size = feat.size()
    assert len(size) == 4, f"Expected 4D tensor (B, C, H, W), got {size}"
    b, c = size[:2]
    feat_var = feat.view(b, c, -1).var(dim=2, keepdim=True) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2, keepdim=True).view(b, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
    """
    Performs Adaptive Instance Normalization:
    AdaIN(x, y) = sigma(y) * ((x - mu(x)) / sigma(x)) + mu(y)
    """
    assert content_feat.size()[:2] == style_feat.size()[:2], \
        f"Content ({content_feat.size()}) and Style ({style_feat.size()}) must have same batch size and channel count."

    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class AdaIN(nn.Module):
    """
    AdaIN module with continuous style strength parameter (alpha in [0.0, 1.0]).
    Alpha = 0.0 -> pure content feature (no style)
    Alpha = 1.0 -> maximum style feature
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        content_feat: torch.Tensor,
        style_feat: torch.Tensor,
        alpha: float = 1.0
    ) -> torch.Tensor:
        """
        Computes interpolated target feature:
        t(alpha) = alpha * AdaIN(fc, fs) + (1 - alpha) * fc
        """
        adain_feat = adaptive_instance_normalization(content_feat, style_feat)
        if alpha == 1.0:
            return adain_feat
        elif alpha == 0.0:
            return content_feat
        else:
            return alpha * adain_feat + (1.0 - alpha) * content_feat
