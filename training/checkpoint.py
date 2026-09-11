"""
Checkpoint saving and loading utilities.
Supports resumable training by saving model state, optimizer state, step counter, and validation metrics.
"""

import os
import torch
import torch.nn as nn
from typing import Optional, Dict, Any


def save_checkpoint(
    decoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    metrics: Dict[str, float],
    checkpoint_dir: str = "c:/art/checkpoints",
    filename: str = "latest.pth"
) -> str:
    """Saves a checkpoint containing model weights, optimizer state, and training metadata."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)

    state = {
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "metrics": metrics
    }
    torch.save(state, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    decoder: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """Restores decoder and optimizer states from a checkpoint file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "decoder" in state:
        decoder.load_state_dict(state["decoder"])
    else:
        decoder.load_state_dict(state)

    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])

    epoch = state.get("epoch", 0)
    step = state.get("step", 0)
    metrics = state.get("metrics", {})

    print(f"[Checkpoint] Successfully loaded from {checkpoint_path} (Epoch {epoch}, Step {step})")
    return {"epoch": epoch, "step": step, "metrics": metrics}
