"""
Utilities for loading pre-trained style transfer models and weights.
Supports downloading canonical AdaIN decoder weights or initializing training from scratch.
"""

import os
import urllib.request
import torch
import torch.nn as nn
from models.decoder import Decoder


def initialize_weights(model: nn.Module):
    """Initializes convolutional layers using Kaiming normal initialization."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def load_pretrained_decoder(
    checkpoint_path: str = "c:/art/checkpoints/decoder_latest.pth",
    device: torch.device = torch.device("cpu")
) -> Decoder:
    """
    Loads trained decoder weights if present; otherwise initializes weights cleanly.
    """
    decoder = Decoder().to(device)

    if os.path.exists(checkpoint_path):
        print(f"[Model] Loading decoder checkpoint from: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "decoder" in state:
            decoder.load_state_dict(state["decoder"])
        elif "model_state_dict" in state:
            decoder.load_state_dict(state["model_state_dict"])
        else:
            if any(k.startswith("decoder.") for k in state.keys()):
                decoder.load_state_dict(state)
            else:
                # Canonical AdaIN decoder weights format ('1.weight', '5.weight', ...)
                decoder.decoder.load_state_dict(state)
    else:
        print("[Model] Initializing decoder with Kaiming normal weights.")
        initialize_weights(decoder)

    return decoder
