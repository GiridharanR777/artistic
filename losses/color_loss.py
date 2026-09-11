"""
Total Variation (TV) and Color Consistency Losses.
Eliminates high-frequency pixel artifacts, salt-and-pepper noise, and unnatural color shifts.
"""

import torch
import torch.nn as nn
from data.preprocessing import rgb_to_ycbcr


class TotalVariationLoss(nn.Module):
    """
    Penalizes high-frequency noise and sudden pixel discontinuities in the generated image.
    Enforces smooth color gradients while preserving artistic brush strokes.
    """

    def __init__(self):
        super().__init__()

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        b, c, h, w = img.size()
        tv_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).sum()
        tv_w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).sum()
        return (tv_h + tv_w) / (b * c * h * w)


class ColorConsistencyLoss(nn.Module):
    """
    Measures deviation in chrominance channels (Cb, Cr) between output and content images.
    Encourages the model to retain natural content color palettes when enabled.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, output: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        out_ycbcr = rgb_to_ycbcr(output.clamp(0, 1))
        cnt_ycbcr = rgb_to_ycbcr(content.clamp(0, 1))

        # Chrominance difference: channels 1 and 2
        cb_diff = self.mse(out_ycbcr[:, 1:2, :, :], cnt_ycbcr[:, 1:2, :, :])
        cr_diff = self.mse(out_ycbcr[:, 2:3, :, :], cnt_ycbcr[:, 2:3, :, :])
        return cb_diff + cr_diff
