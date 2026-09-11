"""
Comparison generator for Artistic Style Transfer.
Produces visual comparison grids (Strength Sweep, Multi-Style Matrix, Color Preservation Comparison)
and computes quantitative metrics.
"""

import os
import sys
import glob
import argparse
from PIL import Image, ImageDraw, ImageFont

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from inference.stylize import stylize, get_model
from data.preprocessing import load_image
from evaluation.metrics import evaluate_transfer_pair


def create_labeled_grid(images: list, labels: list, output_path: str, thumb_size: int = 256):
    """Combines a list of PIL Images horizontally with clear text labels."""
    font = ImageFont.load_default()
    label_height = 24
    total_w = thumb_size * len(images)
    total_h = thumb_size + label_height

    grid = Image.new("RGB", (total_w, total_h), color=(25, 25, 25))
    draw = ImageDraw.Draw(grid)

    for i, (img, label) in enumerate(zip(images, labels)):
        img_resized = img.resize((thumb_size, thumb_size))
        x_offset = i * thumb_size
        # Paste image
        grid.paste(img_resized, (x_offset, 0))
        # Draw label
        draw.text((x_offset + 8, thumb_size + 4), label, fill=(240, 240, 240), font=font)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    grid.save(output_path)
    print(f"[Comparison] Saved grid to: {output_path}")
    return grid


def run_strength_sweep_comparison(
    content_path: str,
    style_path: str,
    output_dir: str = "c:/art/outputs/comparisons",
    strengths: list = [0.0, 0.25, 0.50, 0.75, 1.0],
    image_size: int = 256
):
    """
    Generates a continuous strength sweep comparison:
    Original Content -> Style @ 0.0 -> Style @ 0.25 -> Style @ 0.50 -> Style @ 0.75 -> Style @ 1.0
    """
    content_img = load_image(content_path, image_size=image_size, crop=False)
    style_img = load_image(style_path, image_size=image_size, crop=False)

    images = [content_img, style_img]
    labels = ["Original Content", "Target Style"]

    model = get_model()
    print("\n--- Generating Style Strength Sweep ---")
    for alpha in strengths:
        print(f"Generating strength alpha = {alpha:.2f}...")
        result = stylize(
            content_image=content_img,
            style_image=style_img,
            strength=alpha,
            image_size=image_size
        )
        images.append(result)
        labels.append(f"Strength @ {alpha:.2f}")

        # Metrics
        metrics = evaluate_transfer_pair(result, content_img, style_img, model.encoder)
        print(f"  Alpha {alpha:.2f} -> SSIM: {metrics['SSIM (Content Preservation)']}, "
              f"ContentDist: {metrics['Content Feature Distance']}, "
              f"StyleDist: {metrics['Style Feature Distance']}")

    out_file = os.path.join(output_dir, "strength_sweep_comparison.png")
    create_labeled_grid(images, labels, out_file, thumb_size=image_size)


def run_multi_style_comparison(
    content_path: str,
    style_paths: list,
    output_dir: str = "c:/art/outputs/comparisons",
    strength: float = 0.75,
    image_size: int = 256
):
    """
    Generates a multi-style comparison for a single content image across diverse artistic styles.
    """
    content_img = load_image(content_path, image_size=image_size, crop=False)
    images = [content_img]
    labels = ["Original Content"]

    print("\n--- Generating Multi-Style Comparison ---")
    for s_path in style_paths:
        s_name = os.path.basename(os.path.dirname(s_path)) or os.path.splitext(os.path.basename(s_path))[0]
        style_img = load_image(s_path, image_size=image_size, crop=False)
        result = stylize(
            content_image=content_img,
            style_image=style_img,
            strength=strength,
            image_size=image_size
        )
        images.append(result)
        labels.append(f"{s_name} ({strength:.2f})")

    out_file = os.path.join(output_dir, "multi_style_comparison.png")
    create_labeled_grid(images, labels, out_file, thumb_size=image_size)


def run_color_preservation_comparison(
    content_path: str,
    style_path: str,
    output_dir: str = "c:/art/outputs/comparisons",
    strength: float = 0.75,
    image_size: int = 256
):
    """
    Compares standard RGB artistic style transfer vs Luminance-only Color Preserving transfer.
    """
    content_img = load_image(content_path, image_size=image_size, crop=False)
    style_img = load_image(style_path, image_size=image_size, crop=False)

    standard_result = stylize(
        content_image=content_img,
        style_image=style_img,
        strength=strength,
        color_preserve=False,
        image_size=image_size
    )

    color_preserved_result = stylize(
        content_image=content_img,
        style_image=style_img,
        strength=strength,
        color_preserve=True,
        image_size=image_size
    )

    images = [content_img, style_img, standard_result, color_preserved_result]
    labels = ["Original Content", "Style Image", f"Standard RGB ({strength:.2f})", f"Color-Preserved ({strength:.2f})"]

    out_file = os.path.join(output_dir, "color_preservation_comparison.png")
    create_labeled_grid(images, labels, out_file, thumb_size=image_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Visual Style Transfer Comparisons")
    parser.add_argument("--content", type=str, required=True, help="Path to content image")
    parser.add_argument("--style", type=str, required=True, help="Path to style image")
    parser.add_argument("--size", type=int, default=256, help="Image resolution")
    args = parser.parse_args()

    run_strength_sweep_comparison(args.content, args.style, image_size=args.size)
    run_color_preservation_comparison(args.content, args.style, image_size=args.size)
