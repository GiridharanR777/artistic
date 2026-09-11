"""
Style Loss for Artistic Style Transfer.
Matches mean and standard deviation statistics across multi-scale VGG-19 feature layers:
relu1_1, relu2_1, relu3_1, relu4_1.
"""

import torch
import torch.nn as nn
from typing import Dict
from models.adain import calc_mean_std


class StyleLoss(nn.Module):
    """
    Computes style loss across multi-layer VGG feature representations:
    L_style = sum_i ( || mu(phi_i(out)) - mu(phi_i(style)) ||_2^2 + || sigma(phi_i(out)) - sigma(phi_i(style)) ||_2^2 )
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        output_features: Dict[str, torch.Tensor],
        style_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        loss = 0.0
        for layer_name in output_features.keys():
            out_feat = output_features[layer_name]
            stl_feat = style_features[layer_name]

            out_mean, out_std = calc_mean_std(out_feat)
            stl_mean, stl_std = calc_mean_std(stl_feat)

            loss += self.mse(out_mean, stl_mean) + self.mse(out_std, stl_std)
        return loss
