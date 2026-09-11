"""
Structural Similarity (SSIM) Loss.
Enforces edge integrity, face contours, and geometric layout consistency between
the stylized output and the original content image.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_window(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.tensor([
        math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int) -> torch.Tensor:
    _1d_window = gaussian_window(window_size, 1.5).unsqueeze(1)
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2d_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True
) -> torch.Tensor:
    """Computes differentiable SSIM between two image batches in [0, 1]."""
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device).type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(1).mean(1).mean(1)


class StructuralLoss(nn.Module):
    """
    Structural Loss based on SSIM:
    L_struct = 1 - SSIM(Output, Content)
    """

    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size

    def forward(self, output: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        # Scale to [0, 1] range if needed
        out_clamped = output.clamp(0, 1)
        cnt_clamped = content.clamp(0, 1)
        return 1.0 - ssim(out_clamped, cnt_clamped, window_size=self.window_size)
