"""
PyTorch Dataset and DataLoader definitions for COCO content and WikiArt style images.
Provides configurable subset sizes, train/validation split, and paired content-style sampling.
"""

import os
import glob
import random
from typing import List, Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from data.preprocessing import get_transform, is_valid_image


class ContentDataset(Dataset):
    """Dataset for MS COCO 2017 content images."""

    def __init__(
        self,
        content_dir: str = "c:/art/val2017",
        max_samples: Optional[int] = 500,
        image_size: int = 256,
        is_training: bool = True,
        split: str = "train",
        split_ratio: float = 0.85,
        seed: int = 42
    ):
        super().__init__()
        self.content_dir = content_dir
        self.image_size = image_size
        self.is_training = is_training
        self.transform = get_transform(image_size, is_training)

        all_files = sorted(glob.glob(os.path.join(content_dir, "*.jpg")))
        if not all_files:
            all_files = sorted(glob.glob(os.path.join(content_dir, "*.*")))

        # Deterministic shuffle for reproducible train/val splits
        rng = random.Random(seed)
        shuffled = list(all_files)
        rng.shuffle(shuffled)

        if max_samples and max_samples < len(shuffled):
            shuffled = shuffled[:max_samples]

        split_idx = int(len(shuffled) * split_ratio)
        if split == "train":
            self.files = shuffled[:split_idx]
        elif split in ("val", "validation"):
            self.files = shuffled[split_idx:]
        else:
            self.files = shuffled

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                return self.transform(img)
        except Exception:
            # Fallback to random alternate if image is corrupt
            alt_idx = (idx + 1) % len(self.files)
            with Image.open(self.files[alt_idx]) as img:
                return self.transform(img.convert("RGB"))


class StyleDataset(Dataset):
    """Dataset for WikiArt style images organized by category subfolders."""

    def __init__(
        self,
        style_dir: str = "c:/art/wikiart_cache",
        selected_styles: Optional[List[str]] = None,
        image_size: int = 256,
        is_training: bool = True,
        split: str = "train",
        split_ratio: float = 0.85,
        seed: int = 42
    ):
        super().__init__()
        self.style_dir = style_dir
        self.image_size = image_size
        self.is_training = is_training
        self.transform = get_transform(image_size, is_training)

        self.samples = []  # List of (file_path, style_category)
        if os.path.exists(style_dir):
            categories = os.listdir(style_dir)
            if selected_styles:
                categories = [c for c in categories if c in selected_styles]

            rng = random.Random(seed)
            for cat in categories:
                cat_path = os.path.join(style_dir, cat)
                if os.path.isdir(cat_path):
                    files = sorted(glob.glob(os.path.join(cat_path, "*.*")))
                    cat_samples = [f for f in files if is_valid_image(f)]
                    rng.shuffle(cat_samples)
                    split_idx = int(len(cat_samples) * split_ratio)
                    if split == "train":
                        selected = cat_samples[:split_idx]
                    elif split in ("val", "validation"):
                        selected = cat_samples[split_idx:]
                    else:
                        selected = cat_samples
                    for f in selected:
                        self.samples.append((f, cat))

    def __len__(self) -> int:
        return max(1, len(self.samples))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        if not self.samples:
            # Placeholder tensor if empty
            return torch.zeros(3, self.image_size, self.image_size), "unknown"
        path, cat = self.samples[idx % len(self.samples)]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                return self.transform(img), cat
        except Exception:
            alt_idx = (idx + 1) % len(self.samples)
            path, cat = self.samples[alt_idx]
            with Image.open(path) as img:
                return self.transform(img.convert("RGB")), cat


class PairedStyleTransferDataset(Dataset):
    """
    Pairs content images with randomly sampled style images for end-to-end training.
    """

    def __init__(
        self,
        content_dataset: ContentDataset,
        style_dataset: StyleDataset
    ):
        super().__init__()
        self.content_dataset = content_dataset
        self.style_dataset = style_dataset

    def __len__(self) -> int:
        return len(self.content_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        content_tensor = self.content_dataset[idx]
        style_idx = random.randint(0, len(self.style_dataset) - 1)
        style_tensor, style_category = self.style_dataset[style_idx]
        return content_tensor, style_tensor, style_category


def create_dataloaders(
    content_dir: str = "c:/art/val2017",
    style_dir: str = "c:/art/wikiart_cache",
    selected_styles: Optional[List[str]] = None,
    content_samples: int = 500,
    image_size: int = 256,
    batch_size: int = 2,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """Creates train and validation DataLoaders for paired style transfer."""
    train_content = ContentDataset(content_dir, max_samples=content_samples, image_size=image_size, split="train")
    val_content = ContentDataset(content_dir, max_samples=content_samples, image_size=image_size, split="val")

    train_style = StyleDataset(style_dir, selected_styles=selected_styles, image_size=image_size, split="train")
    val_style = StyleDataset(style_dir, selected_styles=selected_styles, image_size=image_size, split="val")

    train_paired = PairedStyleTransferDataset(train_content, train_style)
    val_paired = PairedStyleTransferDataset(val_content, val_style)

    train_loader = DataLoader(
        train_paired,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    val_loader = DataLoader(
        val_paired,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    return train_loader, val_loader
