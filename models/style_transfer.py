"""
End-to-End Artistic Style Transfer Model.
Combines frozen VGG-19 encoder, AdaIN feature modulation, and trainable mirror decoder.
Supports standard RGB stylization and color-preserving luminance stylization.
"""

import torch
import torch.nn as nn
from models.encoder import VGGEncoder
from models.decoder import Decoder
from models.adain import AdaIN
from models.pretrained import load_pretrained_decoder
from data.preprocessing import rgb_to_ycbcr, ycbcr_to_rgb


class StyleTransferModel(nn.Module):
    """
    Complete Artistic Style Transfer Network.
    Enforces content structure while adopting artistic style features.
    """

    def __init__(self, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device
        self.encoder = VGGEncoder(device=device)
        self.adain = AdaIN()
        self.decoder = load_pretrained_decoder(device=device)

    def forward(
        self,
        content_img: torch.Tensor,
        style_img: torch.Tensor,
        alpha: float = 1.0
    ):
        """
        Forward pass for training or inference.
        Args:
            content_img: (B, 3, H, W) normalized to [0, 1]
            style_img: (B, 3, H, W) normalized to [0, 1]
            alpha: continuous style strength in [0.0, 1.0]
        Returns:
            If in training mode: (stylized_image, target_latent_t)
            If in eval mode: stylized_image
        """
        content_feat = self.encoder.encode(content_img)
        style_feat = self.encoder.encode(style_img)

        # Target modulated feature
        target_feat = self.adain(content_feat, style_feat, alpha=alpha)

        # Decode into stylized RGB image
        stylized_img = self.decoder(target_feat)

        if self.training:
            return stylized_img, target_feat
        return stylized_img

    @torch.no_grad()
    def stylize_color_preserving(
        self,
        content_img: torch.Tensor,
        style_img: torch.Tensor,
        alpha: float = 1.0
    ) -> torch.Tensor:
        """
        Performs artistic style transfer in the luminance channel of YCbCr space.
        Transfers brush strokes, textures, and contrast while preserving original content colors.
        Completely eliminates color cast or blue/yellow hue artifacts.
        """
        content_ycbcr = rgb_to_ycbcr(content_img)
        content_y = content_ycbcr[:, 0:1, :, :].repeat(1, 3, 1, 1)

        style_ycbcr = rgb_to_ycbcr(style_img)
        style_y = style_ycbcr[:, 0:1, :, :].repeat(1, 3, 1, 1)

        # Stylize the luminance map
        stylized_y_rgb = self.forward(content_y, style_y, alpha=alpha)
        stylized_ycbcr = rgb_to_ycbcr(stylized_y_rgb)
        new_y = stylized_ycbcr[:, 0:1, :, :]

        # Combine stylized luminance with original content chrominance
        merged_ycbcr = torch.cat([new_y, content_ycbcr[:, 1:2, :, :], content_ycbcr[:, 2:3, :, :]], dim=1)
        return ycbcr_to_rgb(merged_ycbcr)
