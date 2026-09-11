"""
Artistic Image Steganography Engine.
Provides invisible LSB message embedding and robust decryption for PIL Images.
"""

import os
import struct
import zlib
import hashlib
from typing import Tuple, Optional
import numpy as np
from PIL import Image

# 8-byte Magic Identifier: "ARTSTEG" + version 1
MAGIC_HEADER = b"ARTSTEG\x01"
FLAG_PLAIN = 0x00
FLAG_ENCRYPTED = 0x01


def _derive_keystream(password: str, salt: bytes, length: int) -> bytes:
    """Derives a pseudo-random keystream from password + salt using PBKDF2 HMAC-SHA256."""
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations=10000, dklen=length)
    return derived


def get_capacity(image: Image.Image) -> int:
    """
    Returns the maximum characters of text that can be embedded into the image.
    Each pixel has 3 color channels (R, G, B), each can store 1 bit.
    """
    w, h = image.size
    total_bits = w * h * 3
    # Header overhead: 8 (magic) + 1 (flag) + 16 (salt) + 4 (crc) + 4 (length) = 33 bytes = 264 bits
    header_bits = 33 * 8
    available_bits = total_bits - header_bits
    if available_bits <= 0:
        return 0
    return available_bits // 8


def embed_message(
    image: Image.Image,
    message: str,
    password: Optional[str] = None
) -> Image.Image:
    """
    Embeds a secret text message into a PIL Image using Least Significant Bit (LSB) steganography.

    Args:
        image: Original PIL Image (converted to RGB)
        message: UTF-8 secret text string to embed
        password: Optional encryption passphrase

    Returns:
        New PIL Image with the secret message imperceptibly embedded.
    """
    if not message:
        return image.copy()

    img_rgb = image.convert("RGB")
    w, h = img_rgb.size
    max_bytes = get_capacity(img_rgb)

    payload_bytes = message.encode("utf-8")
    if len(payload_bytes) > max_bytes:
        raise ValueError(
            f"Message size ({len(payload_bytes)} bytes) exceeds image capacity ({max_bytes} bytes)."
        )

    # Compute CRC32 checksum of original plaintext for integrity validation
    crc32 = zlib.crc32(payload_bytes) & 0xFFFFFFFF

    if password:
        flag = FLAG_ENCRYPTED
        salt = os.urandom(16)
        keystream = _derive_keystream(password, salt, len(payload_bytes))
        # XOR encryption
        payload_data = bytes(b ^ k for b, k in zip(payload_bytes, keystream))
    else:
        flag = FLAG_PLAIN
        salt = b"\x00" * 16
        payload_data = payload_bytes

    # Pack full header:
    # 8 bytes magic + 1 byte flag + 16 bytes salt + 4 bytes crc32 + 4 bytes length
    header = MAGIC_HEADER + bytes([flag]) + salt + struct.pack(">II", crc32, len(payload_data))
    full_packet = header + payload_data

    # Convert packet to binary bit array (big-endian bit per byte)
    packet_bits = np.unpackbits(np.frombuffer(full_packet, dtype=np.uint8))

    # Get image array
    arr = np.array(img_rgb, dtype=np.uint8)
    flat_arr = arr.reshape(-1)

    if len(packet_bits) > len(flat_arr):
        raise ValueError("Packet bits exceed total available color channels.")

    # Embed bits into the LSB (clear bit 0, then OR message bit)
    flat_arr[:len(packet_bits)] = (flat_arr[:len(packet_bits)] & 0xFE) | packet_bits

    stego_arr = flat_arr.reshape((h, w, 3))
    return Image.fromarray(stego_arr, mode="RGB")


def extract_message(
    image: Image.Image,
    password: Optional[str] = None
) -> Tuple[bool, str, Optional[str]]:
    """
    Extracts and validates a hidden message from a steganographic PIL Image.

    Args:
        image: PIL Image suspected of containing a hidden message
        password: Optional passphrase if the message was encrypted

    Returns:
        Tuple of (success: bool, message: str, error_detail: Optional[str])
    """
    img_rgb = image.convert("RGB")
    arr = np.array(img_rgb, dtype=np.uint8)
    flat_arr = arr.reshape(-1)

    # Header size = 33 bytes = 264 bits
    header_len_bytes = 33
    header_bits_len = header_len_bytes * 8

    if len(flat_arr) < header_bits_len:
        return False, "", "Image is too small to contain a valid steganography header."

    # Extract header bits
    header_bits = flat_arr[:header_bits_len] & 1
    header_bytes = np.packbits(header_bits).tobytes()

    # Verify magic signature
    if header_bytes[:8] != MAGIC_HEADER:
        return False, "", "No secret message found in this image (missing signature)."

    flag = header_bytes[8]
    salt = header_bytes[9:25]
    crc32, payload_len = struct.unpack(">II", header_bytes[25:33])

    # Validate reasonable payload length
    total_available_bytes = (len(flat_arr) - header_bits_len) // 8
    if payload_len > total_available_bytes or payload_len <= 0:
        return False, "", "Corrupted steganography packet or invalid length."

    # Extract payload bits
    payload_start_bit = header_bits_len
    payload_end_bit = payload_start_bit + (payload_len * 8)

    payload_bits = flat_arr[payload_start_bit:payload_end_bit] & 1
    payload_data = np.packbits(payload_bits).tobytes()

    # Decrypt if needed
    if flag == FLAG_ENCRYPTED:
        if not password:
            return False, "", "This message is encrypted with a passcode. Please provide the key."
        keystream = _derive_keystream(password, salt, payload_len)
        decrypted_bytes = bytes(b ^ k for b, k in zip(payload_data, keystream))
    else:
        decrypted_bytes = payload_data

    # Validate checksum
    calc_crc = zlib.crc32(decrypted_bytes) & 0xFFFFFFFF
    if calc_crc != crc32:
        if flag == FLAG_ENCRYPTED:
            return False, "", "Incorrect passcode or corrupted message."
        return False, "", "Checksum mismatch. The image may have been compressed or modified."

    try:
        message = decrypted_bytes.decode("utf-8")
        return True, message, None
    except UnicodeDecodeError:
        return False, "", "Failed to decode UTF-8 message. Check passcode."


def is_stego_image(image: Image.Image) -> bool:
    """Checks whether an image contains the ARTSTEG signature."""
    try:
        img_rgb = image.convert("RGB")
        arr = np.array(img_rgb, dtype=np.uint8).reshape(-1)
        if len(arr) < 64:
            return False
        header_bits = arr[:64] & 1
        header_bytes = np.packbits(header_bits).tobytes()
        return header_bytes == MAGIC_HEADER
    except Exception:
        return False
