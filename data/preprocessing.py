"""
Image preprocessing and augmentation utilities for artistic style transfer.
Provides robust file validation, RGB conversion, scaling, and color space transformations.
"""

import os
from PIL import Image
import torch
import torchvision.transforms as transforms


def is_valid_image(file_path: str) -> bool:
    """Verifies that an image file exists, is non-empty, and can be opened by PIL."""
    if not os.path.isfile(file_path):
        return False
    if os.path.getsize(file_path) == 0:
        return False
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def load_image(file_path: str, image_size: int = 256, crop: bool = True) -> Image.Image:
    """
    Safely opens an image, converts it to 3-channel RGB, and resizes it.
    If crop is True, performs a center crop after scaling the smaller edge to image_size.
    """
    with Image.open(file_path) as img:
        img = img.convert("RGB")
        if crop:
            # Scale smaller side to image_size and center-crop
            w, h = img.size
            scale = image_size / min(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            left = (new_w - image_size) // 2
            top = (new_h - image_size) // 2
            img = img.crop((left, top, left + image_size, top + image_size))
        else:
            img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
        return img


def get_transform(image_size: int = 256, is_training: bool = True):
    """Returns torchvision transforms converting PIL images to normalized PyTorch tensors."""
    transform_list = []
    if is_training:
        transform_list.extend([
            transforms.Resize(int(image_size * 1.15), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
        ])
    else:
        transform_list.extend([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
        ])
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Converts a (3, H, W) or (1, 3, H, W) PyTorch tensor in [0, 1] to a PIL Image."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    tensor = tensor.detach().cpu().clamp(0, 1)
    to_pil = transforms.ToPILImage()
    return to_pil(tensor)


def rgb_to_ycbcr(tensor: torch.Tensor) -> torch.Tensor:
    """Converts a [0, 1] RGB tensor (B, 3, H, W) to YCbCr."""
    r = tensor[:, 0:1, :, :]
    g = tensor[:, 1:2, :, :]
    b = tensor[:, 2:3, :, :]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return torch.cat([y, cb, cr], dim=1)


def ycbcr_to_rgb(tensor: torch.Tensor) -> torch.Tensor:
    """Converts a YCbCr tensor (B, 3, H, W) to [0, 1] RGB."""
    y = tensor[:, 0:1, :, :]
    cb = tensor[:, 1:2, :, :] - 0.5
    cr = tensor[:, 2:3, :, :] - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return torch.cat([r, g, b], dim=1).clamp(0, 1)
