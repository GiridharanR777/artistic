"""
Central cross-platform path management and model weights auto-downloader.
Ensures the project runs seamlessly on Windows, Linux, Mac, Streamlit Cloud, and Docker.
"""

import os
import urllib.request

# Base project directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
CONFIGS_DIR = os.path.join(BASE_DIR, "configs")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
WIKIART_CACHE_DIR = os.path.join(BASE_DIR, "wikiart_cache")
COCO_DIR = os.path.join(BASE_DIR, "val2017")

VGG_WEIGHTS_PATH = os.path.join(CHECKPOINTS_DIR, "vgg_normalised.pth")
DECODER_WEIGHTS_PATH = os.path.join(CHECKPOINTS_DIR, "decoder.pth")

VGG_WEIGHTS_URL = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/vgg_normalised.pth"
DECODER_WEIGHTS_URL = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/decoder.pth"


def ensure_weights_exist():
    """
    Checks if canonical weights exist locally; if not, automatically downloads them.
    Allows zero-configuration cloud deployment (Streamlit Community Cloud / HuggingFace Spaces).
    """
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    if not os.path.exists(DECODER_WEIGHTS_PATH) or os.path.getsize(DECODER_WEIGHTS_PATH) == 0:
        print("[Setup] Downloading pre-trained decoder.pth (14 MB)...")
        urllib.request.urlretrieve(DECODER_WEIGHTS_URL, DECODER_WEIGHTS_PATH)
        print("[Setup] decoder.pth downloaded.")

    if not os.path.exists(VGG_WEIGHTS_PATH) or os.path.getsize(VGG_WEIGHTS_PATH) == 0:
        print("[Setup] Downloading normalized VGG-19 encoder weights (80 MB)...")
        urllib.request.urlretrieve(VGG_WEIGHTS_URL, VGG_WEIGHTS_PATH)
        print("[Setup] vgg_normalised.pth downloaded.")
