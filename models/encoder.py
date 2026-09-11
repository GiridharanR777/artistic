"""
Normalized VGG-19 Feature Extractor for Artistic Style Transfer.
Matches the exact normalized VGG architecture and weights from vgg_normalised.pth.
Extracts intermediate multi-scale feature maps: relu1_1, relu2_1, relu3_1, relu4_1.
"""

import os
import torch
import torch.nn as nn


from paths import VGG_WEIGHTS_PATH


class VGGEncoder(nn.Module):
    """
    Fixed VGG-19 feature encoder loaded with canonical normalized weights.
    Layer 0 (Conv2d 3->3 1x1) performs the internal color/dynamic range normalization.
    Takes input RGB tensors directly in range [0, 1].
    """

    def __init__(
        self,
        weights_path: str = None,
        device: torch.device = torch.device("cpu")
    ):
        super().__init__()
        self.device = device
        if weights_path is None:
            weights_path = VGG_WEIGHTS_PATH

        vgg = nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),  # relu1-1 (idx 3)
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),  # relu1-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),  # relu2-1 (idx 10)
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),  # relu2-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),  # relu3-1 (idx 17)
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU()   # relu4-1 (idx 30)
        )

        if not os.path.exists(weights_path) or os.path.getsize(weights_path) < 70_000_000:
            from paths import ensure_weights_exist
            ensure_weights_exist()

        if os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=device, weights_only=False)
            vgg.load_state_dict(state, strict=False)
        else:
            raise RuntimeError(f"Required normalized VGG weights not found at {weights_path}")

        enc_layers = list(vgg.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4]).to(device)   # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11]).to(device)  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18]).to(device) # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31]).to(device) # relu3_1 -> relu4_1

        # Freeze encoder parameters
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts relu4_1 feature tensor."""
        h1 = self.enc_1(x)
        h2 = self.enc_2(h1)
        h3 = self.enc_3(h2)
        return self.enc_4(h3)

    def forward(self, x: torch.Tensor):
        """Extracts multi-scale features for multi-layer style loss."""
        h1 = self.enc_1(x)
        h2 = self.enc_2(h1)
        h3 = self.enc_3(h2)
        h4 = self.enc_4(h3)
        return {
            "relu1_1": h1,
            "relu2_1": h2,
            "relu3_1": h3,
            "relu4_1": h4
        }
