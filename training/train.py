"""
Main training script for the Artistic Style Transfer model.
Coordinates dataset loading, optimization loop, loss logging, validation, and checkpointing.
"""

import os
import sys
import argparse
import yaml
import torch

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch.optim import Adam
from data.download_dataset import download_style_subset
from data.dataset import create_dataloaders
from models.style_transfer import StyleTransferModel
from models.pretrained import load_pretrained_decoder
from losses.total_loss import TotalStyleTransferLoss
from training.checkpoint import save_checkpoint, load_checkpoint
from training.validate import evaluate_validation


def load_config(config_path: str = "c:/art/configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train(config_path: str = "c:/art/configs/config.yaml", max_steps: int = 0):
    cfg = load_config(config_path)
    device = torch.device("cpu")
    print(f"[Training] Running on device: {device}")

    # 1. Ensure style images are cached
    ds_cfg = cfg["dataset"]
    print(f"[Training] Ensuring WikiArt style images are ready in {ds_cfg['wikiart_cache_dir']}...")
    download_style_subset(
        selected_styles=ds_cfg["selected_styles"],
        samples_per_category=ds_cfg["style_samples_per_category"],
        output_dir=ds_cfg["wikiart_cache_dir"]
    )

    # 2. Create DataLoaders
    print(f"[Training] Creating DataLoaders (COCO samples: {ds_cfg['content_subset_size']})...")
    train_loader, val_loader = create_dataloaders(
        content_dir=ds_cfg["content_dir"],
        style_dir=ds_cfg["wikiart_cache_dir"],
        selected_styles=ds_cfg["selected_styles"],
        content_samples=ds_cfg["content_subset_size"],
        image_size=ds_cfg["image_size"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"]
    )
    print(f"[Training] Loaded {len(train_loader)} training batches, {len(val_loader)} validation batches.")

    # 3. Model setup
    model = StyleTransferModel(device=device)
    # Check if a starting checkpoint is specified or present
    resume_path = cfg["training"].get("resume_checkpoint", "")
    if resume_path and os.path.exists(resume_path):
        load_checkpoint(resume_path, model.decoder, device=device)

    # Only train the decoder!
    optimizer = Adam(
        model.decoder.parameters(),
        lr=cfg["training"]["learning_rate"]
    )

    # 4. Multi-objective loss setup
    loss_cfg = cfg["losses"]
    criterion = TotalStyleTransferLoss(
        encoder=model.encoder,
        content_weight=loss_cfg["content_weight"],
        style_weight=loss_cfg["style_weight"],
        structural_weight=loss_cfg["structural_weight"],
        tv_weight=loss_cfg["tv_weight"]
    ).to(device)

    # 5. Training loop
    model.train()
    epochs = cfg["training"]["epochs"]
    save_interval = cfg["training"]["save_interval_steps"]
    checkpoint_dir = cfg["training"]["checkpoint_dir"]
    best_loss = float("inf")
    global_step = 0

    print("[Training] Starting training loop...")
    for epoch in range(1, epochs + 1):
        for batch_idx, (content_imgs, style_imgs, style_names) in enumerate(train_loader):
            content_imgs = content_imgs.to(device)
            style_imgs = style_imgs.to(device)

            optimizer.zero_grad()

            # Forward pass: model outputs stylized image and target feature
            stylized_imgs, target_feats = model(content_imgs, style_imgs, alpha=1.0)

            # Compute multi-objective loss
            loss, metrics = criterion(stylized_imgs, content_imgs, style_imgs, target_feats)

            # Backprop and optimize decoder
            loss.backward()
            optimizer.step()

            global_step += 1

            if global_step % 10 == 0 or global_step == 1:
                print(
                    f"Epoch [{epoch}/{epochs}] Step [{global_step}] "
                    f"Total Loss: {metrics['total_loss']:.4f} "
                    f"(Content: {metrics['content_loss']:.4f}, "
                    f"Style: {metrics['style_loss']:.4f}, "
                    f"Struct: {metrics['struct_loss']:.4f}, "
                    f"TV: {metrics['tv_loss']:.4f})"
                )

            if global_step % save_interval == 0:
                save_checkpoint(
                    model.decoder, optimizer, epoch, global_step, metrics,
                    checkpoint_dir=checkpoint_dir, filename=f"step_{global_step:05d}.pth"
                )
                save_checkpoint(
                    model.decoder, optimizer, epoch, global_step, metrics,
                    checkpoint_dir=checkpoint_dir, filename="latest.pth"
                )

            if max_steps and global_step >= max_steps:
                print(f"[Training] Reached verification max_steps ({max_steps}). Finishing.")
                break

        if max_steps and global_step >= max_steps:
            break

        # Epoch Validation
        print(f"[Training] Running validation for Epoch {epoch}...")
        val_metrics = evaluate_validation(model, val_loader, criterion, device=device, epoch=epoch)
        print(f"[Validation] Epoch {epoch} - Val Loss: {val_metrics['val_loss']:.4f}, SSIM: {val_metrics['val_ssim']:.4f}")

        # Save latest and best checkpoints
        save_checkpoint(
            model.decoder, optimizer, epoch, global_step, val_metrics,
            checkpoint_dir=checkpoint_dir, filename="latest.pth"
        )
        if val_metrics["val_loss"] < best_loss:
            best_loss = val_metrics["val_loss"]
            save_checkpoint(
                model.decoder, optimizer, epoch, global_step, val_metrics,
                checkpoint_dir=checkpoint_dir, filename="best.pth"
            )
            print(f"[Checkpoint] New best model saved with Val Loss {best_loss:.4f}!")

    print("[Training] Completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Artistic Style Transfer Model")
    parser.add_argument("--config", type=str, default="c:/art/configs/config.yaml", help="Path to config file")
    parser.add_argument("--max_steps", type=int, default=0, help="Optional maximum steps for quick pipeline verification")
    args = parser.parse_args()

    train(config_path=args.config, max_steps=args.max_steps)
