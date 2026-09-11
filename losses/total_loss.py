"""
Multi-Objective Loss Coordinator for Artistic Style Transfer.
Balances content preservation, style feature transfer, structural edge integrity,
and total variation regularization using configurable weights.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
from models.encoder import VGGEncoder
from losses.content_loss import ContentLoss
from losses.style_loss import StyleLoss
from losses.structural_loss import StructuralLoss
from losses.color_loss import TotalVariationLoss, ColorConsistencyLoss


class TotalStyleTransferLoss(nn.Module):
    """
    Computes weighted multi-objective loss:
    Total Loss = lambda_c * L_content + lambda_s * L_style + lambda_struct * L_struct + lambda_tv * L_tv
    """

    def __init__(
        self,
        encoder: VGGEncoder,
        content_weight: float = 1.0,
        style_weight: float = 10.0,
        structural_weight: float = 1.0,
        tv_weight: float = 1e-5,
        color_weight: float = 0.0
    ):
        super().__init__()
        self.encoder = encoder
        self.content_loss_fn = ContentLoss()
        self.style_loss_fn = StyleLoss()
        self.struct_loss_fn = StructuralLoss()
        self.tv_loss_fn = TotalVariationLoss()
        self.color_loss_fn = ColorConsistencyLoss()

        self.content_weight = content_weight
        self.style_weight = style_weight
        self.structural_weight = structural_weight
        self.tv_weight = tv_weight
        self.color_weight = color_weight

    def forward(
        self,
        output_img: torch.Tensor,
        content_img: torch.Tensor,
        style_img: torch.Tensor,
        target_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the total loss and individual metric breakdowns.
        """
        # Extract features for output and style
        out_feats = self.encoder(output_img)
        style_feats = self.encoder(style_img)

        # 1. Content Loss: match relu4_1 features against target AdaIN feature t
        loss_content = self.content_loss_fn(out_feats["relu4_1"], target_feat)

        # 2. Style Loss: match mean and std across relu1_1, relu2_1, relu3_1, relu4_1
        loss_style = self.style_loss_fn(out_feats, style_feats)

        # 3. Structural Loss: SSIM against original content
        loss_struct = self.struct_loss_fn(output_img, content_img) if self.structural_weight > 0 else torch.tensor(0.0)

        # 4. Total Variation Loss: remove noise and pixel ringing
        loss_tv = self.tv_loss_fn(output_img) if self.tv_weight > 0 else torch.tensor(0.0)

        # 5. Optional Color consistency loss
        loss_color = self.color_loss_fn(output_img, content_img) if self.color_weight > 0 else torch.tensor(0.0)

        total_loss = (
            self.content_weight * loss_content +
            self.style_weight * loss_style +
            self.structural_weight * loss_struct +
            self.tv_weight * loss_tv +
            self.color_weight * loss_color
        )

        metrics = {
            "total_loss": total_loss.item(),
            "content_loss": loss_content.item(),
            "style_loss": loss_style.item(),
            "struct_loss": loss_struct.item() if isinstance(loss_struct, torch.Tensor) else 0.0,
            "tv_loss": loss_tv.item() if isinstance(loss_tv, torch.Tensor) else 0.0,
        }

        return total_loss, metrics
