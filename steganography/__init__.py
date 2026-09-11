"""
Artistic Image Steganography Package.
Enables hiding secret messages in stylized images and decoding them with optional encryption.
"""

from .codec import (
    embed_message,
    extract_message,
    get_capacity,
    is_stego_image,
)

__all__ = [
    "embed_message",
    "extract_message",
    "get_capacity",
    "is_stego_image",
]
