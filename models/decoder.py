"""
Inverted VGG Decoder Network.
Decodes relu4_1 feature representations back into RGB image space.
Uses reflection padding and nearest-neighbor upsampling to avoid checkerboard artifacts.
"""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    Symmetrical convolutional decoder mirroring VGG-19 up to relu4_1.
    Input: (B, 512, H/8, W/8) feature maps.
    Output: (B, 3, H, W) reconstructed RGB images in range [0, 1].
    """

    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            # Stage 4: 512 -> 256
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),

            # Stage 3: 256 -> 128
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),

            # Stage 2: 128 -> 64
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),

            # Stage 1: 64 -> 3
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, kernel_size=3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs image from feature map x.
        Clamps output to valid RGB range [0, 1].
        """
        out = self.decoder(x)
        return torch.clamp(out, 0.0, 1.0)
