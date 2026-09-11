"""
Pipeline Verification Script (Stage 1).
Tests dataset loading, model forward pass, multi-objective loss computation,
backward pass gradient flow, inference API, continuous style strength sweep,
and evaluation metrics calculation.
"""

import os
import sys
import glob
import torch

# Ensure project root is in sys.path
sys.path.insert(0, "c:/art")

from data.download_dataset import download_style_subset
from data.dataset import create_dataloaders
from models.style_transfer import StyleTransferModel
from losses.total_loss import TotalStyleTransferLoss
from inference.stylize import stylize
from evaluation.generate_comparison import run_strength_sweep_comparison, run_color_preservation_comparison


def run_pipeline_verification():
    print("==========================================================")
    print("          STAGE 1: PIPELINE VERIFICATION RUN              ")
    print("==========================================================")
    device = torch.device("cpu")

    # Step 1: Download a tiny style subset (3 categories, 5 samples each)
    print("\n[Step 1/5] Downloading tiny WikiArt sample subset...")
    download_style_subset(
        selected_styles=["Impressionism", "Cubism", "Ukiyo_e"],
        samples_per_category=5,
        output_dir="c:/art/wikiart_cache"
    )

    # Step 2: Test DataLoaders
    print("\n[Step 2/5] Initializing Paired DataLoaders...")
    train_loader, val_loader = create_dataloaders(
        content_dir="c:/art/val2017",
        style_dir="c:/art/wikiart_cache",
        selected_styles=["Impressionism", "Cubism", "Ukiyo_e"],
        content_samples=20,
        image_size=256,
        batch_size=2
    )
    content_b, style_b, styles = next(iter(train_loader))
    print(f"  Content batch shape: {content_b.shape}")
    print(f"  Style batch shape:   {style_b.shape}")
    print(f"  Sample styles:       {styles}")
    assert content_b.shape == (2, 3, 256, 256), f"Unexpected shape {content_b.shape}"

    # Step 3: Test Model Forward Pass
    print("\n[Step 3/5] Testing Model Forward Pass...")
    model = StyleTransferModel(device=device)
    model.train()
    stylized_b, target_feat = model(content_b, style_b, alpha=0.75)
    print(f"  Stylized output shape: {stylized_b.shape}")
    print(f"  Target feature shape:  {target_feat.shape}")
    assert stylized_b.shape == (2, 3, 256, 256), "Output shape mismatch!"

    # Step 4: Test Multi-Objective Loss & Backward Pass
    print("\n[Step 4/5] Testing Multi-Objective Loss and Gradient Flow...")
    criterion = TotalStyleTransferLoss(
        encoder=model.encoder,
        content_weight=1.0,
        style_weight=10.0,
        structural_weight=1.0,
        tv_weight=1e-5
    )
    loss, metrics = criterion(stylized_b, content_b, style_b, target_feat)
    print(f"  Total Loss:      {metrics['total_loss']:.4f}")
    print(f"  Content Loss:    {metrics['content_loss']:.4f}")
    print(f"  Style Loss:      {metrics['style_loss']:.4f}")
    print(f"  Structural Loss: {metrics['struct_loss']:.4f}")
    print(f"  TV Loss:         {metrics['tv_loss']:.4f}")

    # Backward pass
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.decoder.parameters() if p.grad is not None)
    print(f"  Decoder gradient norm: {grad_norm:.4f}")
    assert grad_norm > 0, "Gradients failed to flow to decoder parameters!"
    print("  [PASS] Gradient propagation verified successfully.")

    # Step 5: Test Inference & Strength Sweep
    print("\n[Step 5/5] Testing Inference API and Visual Comparisons...")
    coco_samples = sorted(glob.glob("c:/art/val2017/*.jpg"))
    sample_content = coco_samples[0]
    style_samples = sorted(glob.glob("c:/art/wikiart_cache/*/*.jpg"))
    sample_style = style_samples[0]

    print(f"  Using Content: {sample_content}")
    print(f"  Using Style:   {sample_style}")

    # Run single stylization
    out_single = stylize(
        content_image=sample_content,
        style_image=sample_style,
        strength=0.75,
        output_path="c:/art/outputs/single_test.jpg"
    )
    print("  [PASS] Single image inference passed.")

    # Run continuous strength sweep
    run_strength_sweep_comparison(
        content_path=sample_content,
        style_path=sample_style,
        output_dir="c:/art/outputs/comparisons",
        strengths=[0.0, 0.25, 0.50, 0.75, 1.0],
        image_size=256
    )
    print("  [PASS] Strength sweep comparison generated.")

    # Run color preservation comparison
    run_color_preservation_comparison(
        content_path=sample_content,
        style_path=sample_style,
        output_dir="c:/art/outputs/comparisons",
        strength=0.75,
        image_size=256
    )
    print("  [PASS] Color preservation comparison generated.")

    print("\n==========================================================")
    print("     ALL PIPELINE VERIFICATION TESTS PASSED CLEANLY!      ")
    print("==========================================================")


if __name__ == "__main__":
    run_pipeline_verification()
