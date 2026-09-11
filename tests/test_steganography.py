"""
Automated unit tests for Image Steganography module.
"""

import sys
import os
import unittest
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from steganography import (
    embed_message,
    extract_message,
    get_capacity,
    is_stego_image
)


class TestSteganography(unittest.TestCase):

    def setUp(self):
        # Create a sample synthetic 256x256 image with natural-like pixel gradients
        x = np.linspace(0, 255, 256, dtype=np.uint8)
        y = np.linspace(0, 255, 256, dtype=np.uint8)
        xx, yy = np.meshgrid(x, y)
        r = xx
        g = yy
        b = ((xx.astype(int) + yy.astype(int)) // 2).astype(np.uint8)
        rgb = np.stack([r, g, b], axis=-1)
        self.sample_img = Image.fromarray(rgb, mode="RGB")

    def test_user_example_message(self):
        """Test the user's exact example: 'i want to meet u' without password."""
        msg = "i want to meet u"
        stego_img = embed_message(self.sample_img, msg)
        self.assertTrue(is_stego_image(stego_img))

        success, extracted, err = extract_message(stego_img)
        self.assertTrue(success)
        self.assertEqual(extracted, msg)
        self.assertIsNone(err)

    def test_encrypted_message(self):
        """Test message embedding with passcode encryption and decryption."""
        msg = "Secret meeting at 10 PM at the gallery 🎨"
        password = "ArtSecureKey!42"

        stego_img = embed_message(self.sample_img, msg, password=password)
        self.assertTrue(is_stego_image(stego_img))

        # 1. Correct password should succeed
        success, extracted, err = extract_message(stego_img, password=password)
        self.assertTrue(success)
        self.assertEqual(extracted, msg)

        # 2. Wrong password should fail safely
        success_bad, extracted_bad, err_bad = extract_message(stego_img, password="WrongPassword")
        self.assertFalse(success_bad)
        self.assertEqual(extracted_bad, "")
        self.assertIn("passcode", err_bad.lower())

        # 3. Missing password should prompt for key
        success_none, extracted_none, err_none = extract_message(stego_img, password=None)
        self.assertFalse(success_none)
        self.assertIn("passcode", err_none.lower())

    def test_non_stego_image(self):
        """Test that a plain image without hidden message returns clean false."""
        self.assertFalse(is_stego_image(self.sample_img))
        success, extracted, err = extract_message(self.sample_img)
        self.assertFalse(success)
        self.assertEqual(extracted, "")
        self.assertIn("No secret message", err)

    def test_multiline_and_unicode(self):
        """Test long multiline text with special characters."""
        long_msg = "Line 1: Hello!\nLine 2: こんにちは世界\nLine 3: Bonjour le monde 🌍✨"
        stego_img = embed_message(self.sample_img, long_msg)
        success, extracted, err = extract_message(stego_img)
        self.assertTrue(success)
        self.assertEqual(extracted, long_msg)

    def test_imperceptibility_psnr(self):
        """Test that embedding introduces imperceptible changes (PSNR > 50 dB)."""
        msg = "High fidelity steganography test message."
        stego_img = embed_message(self.sample_img, msg)

        arr1 = np.array(self.sample_img, dtype=np.float64)
        arr2 = np.array(stego_img, dtype=np.float64)
        mse = np.mean((arr1 - arr2) ** 2)

        if mse == 0:
            psnr = float("inf")
        else:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))

        print(f"\n[Stego Fidelity] PSNR = {psnr:.2f} dB (MSE = {mse:.6f})")
        self.assertGreater(psnr, 50.0, "PSNR should be greater than 50 dB for imperceptibility.")

    def test_capacity_check(self):
        """Test capacity calculation."""
        cap = get_capacity(self.sample_img)
        # 256 * 256 * 3 = 196,608 bits. 196,608 - 264 = 196,344 bits / 8 = 24,543 bytes.
        self.assertGreater(cap, 24000)


if __name__ == "__main__":
    unittest.main()
