"""
Content Loss for Artistic Style Transfer.
Measures L2 difference between relu4_1 feature maps of stylized output and the target AdaIN feature t.
"""

import torch
import torch.nn as nn


class ContentLoss(nn.Module):
    """
    Computes Euclidean distance in feature space:
    L_content = || f(I_out) - t ||_2^2
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, output_feature: torch.Tensor, target_feature: torch.Tensor) -> torch.Tensor:
        return self.mse(output_feature, target_feature)
