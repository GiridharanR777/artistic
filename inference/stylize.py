"""
Inference API and CLI for Artistic Style Transfer.
Supports arbitrary style images or named style categories, continuous style strength,
and color-preserving transfer mode.
"""

import os
import sys
import glob
import random
import argparse
from typing import Optional, Union
from PIL import Image
import torch

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torchvision.transforms as transforms
from models.style_transfer import StyleTransferModel
from models.pretrained import load_pretrained_decoder
from data.preprocessing import load_image, tensor_to_image
from paths import DECODER_WEIGHTS_PATH, CHECKPOINTS_DIR, WIKIART_CACHE_DIR


# Global model cache to avoid reloading weights repeatedly
_CACHED_MODEL = None


def get_model(
    checkpoint_path: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
    force_reload: bool = False
) -> StyleTransferModel:
    """Returns a cached or newly initialized StyleTransferModel."""
    global _CACHED_MODEL
    if _CACHED_MODEL is not None and checkpoint_path is None and not force_reload:
        return _CACHED_MODEL

    model = StyleTransferModel(device=device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.decoder = load_pretrained_decoder(checkpoint_path, device=device)
    else:
        # Check standard checkpoints using cross-platform paths
        for candidate in [
            DECODER_WEIGHTS_PATH,
            os.path.join(CHECKPOINTS_DIR, "decoder.pth"),
            os.path.join(CHECKPOINTS_DIR, "best.pth"),
            os.path.join(CHECKPOINTS_DIR, "latest.pth"),
            os.path.join(CHECKPOINTS_DIR, "decoder_latest.pth")
        ]:
            if os.path.exists(candidate):
                model.decoder = load_pretrained_decoder(candidate, device=device)
                break

    model.eval()
    _CACHED_MODEL = model
    return model


def resolve_style_image(style: str, wikiart_cache_dir: str = None) -> str:
    """
    Resolves a style parameter to a valid image file path.
    If style is a file path that exists, returns it directly.
    If style is a named category (e.g. 'impressionism', 'cubism'), samples an image from the cache.
    """
    if wikiart_cache_dir is None:
        wikiart_cache_dir = WIKIART_CACHE_DIR
    if os.path.isfile(style):
        return style

    # Try matching category in wikiart_cache
    if os.path.exists(wikiart_cache_dir):
        categories = os.listdir(wikiart_cache_dir)
        # Case-insensitive match
        for cat in categories:
            if cat.lower() == style.lower() or cat.lower().replace("_", "") == style.lower().replace("_", ""):
                cat_dir = os.path.join(wikiart_cache_dir, cat)
                files = [f for f in glob.glob(os.path.join(cat_dir, "*.*")) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
                if files:
                    return random.choice(files)

    raise FileNotFoundError(f"Could not find style image or category matching '{style}'.")


def stylize(
    content_image: Union[str, Image.Image],
    style_image: Optional[Union[str, Image.Image]] = None,
    style: Optional[str] = None,
    strength: float = 0.75,
    color_preserve: bool = False,
    image_size: int = 256,
    output_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
    force_reload: bool = False
) -> Image.Image:
    """
    Main Python Inference Interface for Artistic Style Transfer.

    Args:
        content_image: Path to content image or PIL Image object
        style_image: Path to style image or PIL Image object
        style: Name of an artistic category (e.g. 'impressionism', 'cubism', 'ukiyo_e')
        strength: Style transfer intensity in [0.0, 1.0]. 0.0 = original, 1.0 = full style
        color_preserve: If True, preserves original content colors and applies only brushwork/texture
        image_size: Inference resolution (default 256, can be 512)
        output_path: Optional path to save the resulting image
        checkpoint_path: Optional path to decoder weights checkpoint
        device: Torch device (default CPU)
        force_reload: Force re-instantiation of the model

    Returns:
        PIL Image of the stylized result
    """
    # 1. Resolve style
    if style_image is None and style is not None:
        style_path = resolve_style_image(style)
        style_img = load_image(style_path, image_size=image_size, crop=False)
    elif isinstance(style_image, str):
        style_img = load_image(style_image, image_size=image_size, crop=False)
    elif isinstance(style_image, Image.Image):
        style_img = style_image.resize((image_size, image_size))
    else:
        raise ValueError("Either style_image or style category must be specified.")

    # 2. Resolve content
    if isinstance(content_image, str):
        content_img = load_image(content_image, image_size=image_size, crop=False)
    elif isinstance(content_image, Image.Image):
        content_img = content_image.resize((image_size, image_size))
    else:
        raise ValueError("content_image must be a file path or PIL Image.")

    # 3. Convert to Tensors
    to_tensor = transforms.ToTensor()
    c_tensor = to_tensor(content_img).unsqueeze(0).to(device)
    s_tensor = to_tensor(style_img).unsqueeze(0).to(device)

    # 4. Model inference
    model = get_model(checkpoint_path=checkpoint_path, device=device, force_reload=force_reload)

    with torch.no_grad():
        if color_preserve:
            output_tensor = model.stylize_color_preserving(c_tensor, s_tensor, alpha=strength)
        else:
            output_tensor = model(c_tensor, s_tensor, alpha=strength)

    # 5. Convert back to PIL Image
    result_img = tensor_to_image(output_tensor)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        result_img.save(output_path)
        print(f"[Inference] Saved stylized image to: {output_path}")

    return result_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Artistic Style Transfer Inference")
    parser.add_argument("--content", type=str, required=True, help="Path to content image")
    parser.add_argument("--style", type=str, required=True, help="Path to style image or named style category")
    parser.add_argument("--strength", type=float, default=0.75, help="Style strength alpha in [0.0, 1.0]")
    parser.add_argument("--color_preserve", action="store_true", help="Preserve original content color palette")
    parser.add_argument("--size", type=int, default=256, help="Image resolution")
    parser.add_argument("--output", type=str, default="c:/art/outputs/result.jpg", help="Output save path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom checkpoint path")
    args = parser.parse_args()

    stylize(
        content_image=args.content,
        style=args.style if not os.path.isfile(args.style) else None,
        style_image=args.style if os.path.isfile(args.style) else None,
        strength=args.strength,
        color_preserve=args.color_preserve,
        image_size=args.size,
        output_path=args.output,
        checkpoint_path=args.checkpoint
    )
