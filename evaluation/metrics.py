"""
Evaluation metrics for Artistic Style Transfer.
Provides quantitative evaluation (SSIM, PSNR, Content Feature Distance, Style Feature Distance)
and documents the practical meaning and limitations of each metric.
"""

import math
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from models.encoder import VGGEncoder
from models.adain import calc_mean_std
from losses.structural_loss import ssim


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in dB.
    Limitation: PSNR assumes any pixel deviation is 'noise'. For artistic style transfer,
    artistic strokes and color changes deliberately alter pixels, so low PSNR does NOT mean bad quality.
    It is primarily useful for verifying that alpha=0.0 accurately reconstructs the original.
    """
    mse = F.mse_loss(img1.clamp(0, 1), img2.clamp(0, 1)).item()
    if mse == 0:
        return 100.0
    return 20 * math.log10(1.0 / math.sqrt(mse))


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Computes Structural Similarity Index (SSIM).
    Limitation: Measures structural and contrast preservation. High SSIM (>0.7) confirms
    facial features and geometry are preserved. However, if SSIM is too close to 1.0,
    the style transfer may be too weak.
    """
    return ssim(img1.clamp(0, 1), img2.clamp(0, 1)).item()


def calculate_feature_distances(
    encoder: VGGEncoder,
    output_img: torch.Tensor,
    content_img: torch.Tensor,
    style_img: torch.Tensor
) -> dict:
    """
    Computes semantic content distance and statistical style distance.
    - Content Distance: L2 distance in VGG relu4_1 space. Low distance = semantics preserved.
    - Style Distance: Multi-scale mean & std discrepancy. Low distance = style faithfully transferred.
    """
    with torch.no_grad():
        out_feats = encoder(output_img)
        cnt_feats = encoder(content_img)
        stl_feats = encoder(style_img)

        # Content distance at relu4_1
        c_dist = F.mse_loss(out_feats["relu4_1"], cnt_feats["relu4_1"]).item()

        # Style distance across all 4 layers
        s_dist = 0.0
        for k in out_feats.keys():
            om, os = calc_mean_std(out_feats[k])
            sm, ss = calc_mean_std(stl_feats[k])
            s_dist += (F.mse_loss(om, sm) + F.mse_loss(os, ss)).item()

    return {
        "content_feature_dist": c_dist,
        "style_feature_dist": s_dist
    }


def evaluate_transfer_pair(
    output_img: Image.Image,
    content_img: Image.Image,
    style_img: Image.Image,
    encoder: VGGEncoder,
    device: torch.device = torch.device("cpu")
) -> dict:
    """Computes all metrics for a single (content, style, output) triplet."""
    target_size = output_img.size  # (W, H)
    c_resized = content_img.resize(target_size, Image.Resampling.BILINEAR)
    s_resized = style_img.resize(target_size, Image.Resampling.BILINEAR)

    to_tensor = transforms.ToTensor()
    o_t = to_tensor(output_img).unsqueeze(0).to(device)
    c_t = to_tensor(c_resized).unsqueeze(0).to(device)
    s_t = to_tensor(s_resized).unsqueeze(0).to(device)

    psnr_val = calculate_psnr(o_t, c_t)
    ssim_val = calculate_ssim(o_t, c_t)
    feat_dists = calculate_feature_distances(encoder, o_t, c_t, s_t)

    return {
        "SSIM (Content Preservation)": round(ssim_val, 4),
        "PSNR (Reconstruction dB)": round(psnr_val, 2),
        "Content Feature Distance": round(feat_dists["content_feature_dist"], 4),
        "Style Feature Distance": round(feat_dists["style_feature_dist"], 4)
    }
