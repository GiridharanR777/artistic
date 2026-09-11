"""
Validation routines for assessing style transfer quality, loss convergence, and content preservation.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict
from torchvision.utils import save_image
from losses.total_loss import TotalStyleTransferLoss
from losses.structural_loss import ssim


@torch.no_grad()
def evaluate_validation(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: TotalStyleTransferLoss,
    device: torch.device = torch.device("cpu"),
    save_samples_dir: str = "c:/art/outputs/val_samples",
    epoch: int = 0
) -> Dict[str, float]:
    """
    Evaluates the model on unseen content and style validation images.
    Saves qualitative comparison samples to save_samples_dir.
    """
    model.eval()
    os.makedirs(save_samples_dir, exist_ok=True)

    total_loss = 0.0
    total_ssim = 0.0
    sample_saved = False

    num_batches = len(val_loader)
    if num_batches == 0:
        return {"val_loss": 0.0, "val_ssim": 1.0}

    for i, (content_imgs, style_imgs, style_names) in enumerate(val_loader):
        content_imgs = content_imgs.to(device)
        style_imgs = style_imgs.to(device)

        # Forward pass
        stylized_imgs, target_feats = model(content_imgs, style_imgs, alpha=1.0)
        loss, _ = criterion(stylized_imgs, content_imgs, style_imgs, target_feats)

        total_loss += loss.item()
        batch_ssim = ssim(stylized_imgs.clamp(0, 1), content_imgs.clamp(0, 1)).item()
        total_ssim += batch_ssim

        # Save first batch as visual evaluation sample
        if not sample_saved:
            sample_grid = torch.cat([content_imgs[:2], style_imgs[:2], stylized_imgs[:2]], dim=0)
            sample_path = os.path.join(save_samples_dir, f"epoch_{epoch:03d}_val_preview.png")
            save_image(sample_grid, sample_path, nrow=2, normalize=False)
            sample_saved = True

    avg_loss = total_loss / num_batches
    avg_ssim = total_ssim / num_batches

    model.train()
    return {"val_loss": avg_loss, "val_ssim": avg_ssim}
