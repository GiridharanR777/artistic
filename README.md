# High-Quality Artistic Image Style Transfer

A modular, hardware-aware PyTorch framework for artistic style transfer with strict content preservation and continuous, controllable style strength.

---

## Key Features

1. **Strict Content Preservation**: Deep VGG-19 semantic feature encoding prevents geometric collapse, distorted faces, or broken object contours.
2. **Continuous Style Strength ($\alpha \in [0.0, 1.0]$)**: Mathematical latent-space feature interpolation:
   $$t(\alpha) = \alpha \cdot \text{AdaIN}(f_c, f_s) + (1 - \alpha) \cdot f_c$$
   Smoothly scales from pristine original content ($\alpha=0.0$) to intense artistic rendering ($\alpha=1.0$).
3. **Artifact Prevention & Color Preservation**:
   - Differentiable **Structural SSIM Loss** preserves edges and facial geometry.
   - **Total Variation (TV) Regularization** eliminates high-frequency noise and pixel ringing.
   - **Luminance-Preserving Mode (YCbCr)** applies painterly brushstrokes while guaranteeing zero unnatural color shift or blue/yellow hue tinting.
4. **Targeted On-Demand WikiArt Streaming**: Streams selected styles from Kaggle (`steubk/wikiart`) via `kagglehub` without needing to download all 80,000 images at once.
5. **Hardware-Aware Design**: Optimized for CPU and low VRAM environments with zero out-of-memory risk.

---

## Directory Structure

```text
art/
│
├── configs/
│   └── config.yaml               # Configurable dataset paths, subsets, loss weights
│
├── data/
│   ├── dataset.py                # PyTorch Dataset/DataLoader (COCO & WikiArt)
│   ├── preprocessing.py          # Robust image loader, transforms, YCbCr converters
│   └── download_dataset.py       # KaggleHub targeted downloader for WikiArt categories
│
├── models/
│   ├── encoder.py                # Frozen VGG-19 feature extractor (relu1_1 to relu4_1)
│   ├── decoder.py                # Inverted VGG convolutional decoder
│   ├── adain.py                  # Adaptive Instance Normalization with strength alpha
│   ├── style_transfer.py         # End-to-end model and color-preserving pipeline
│   └── pretrained.py             # Pretrained weights loader and weight initializers
│
├── losses/
│   ├── content_loss.py           # VGG relu4_1 feature MSE
│   ├── style_loss.py             # Multi-scale mean & std statistic alignment
│   ├── structural_loss.py        # SSIM structural preservation loss
│   ├── color_loss.py             # Total variation & color consistency losses
│   └── total_loss.py             # Multi-objective loss coordinator
│
├── training/
│   ├── train.py                  # Training pipeline with checkpointing
│   ├── validate.py               # Validation evaluation loop
│   └── checkpoint.py             # Checkpoint saver/loader
│
├── inference/
│   └── stylize.py                # Python API and CLI for single/batch stylization
│
├── evaluation/
│   ├── metrics.py                # SSIM, PSNR, feature distance calculations
│   └── generate_comparison.py    # Generates visual comparison grids and sweeps
│
├── simple_ui.py                  # Lightweight Streamlit UI for interactive testing
├── outputs/                      # Saved images and comparison grids
├── checkpoints/                  # Saved model checkpoints
└── requirements.txt
```

---

## Quick Start

### 1. Python Inference API

```python
from inference.stylize import stylize

# Stylize with arbitrary style image
result = stylize(
    content_image="c:/art/val2017/000000000139.jpg",
    style_image="c:/art/wikiart_cache/Impressionism/claude-monet_water-lilies.jpg",
    strength=0.65,
    color_preserve=False,
    output_path="c:/art/outputs/result.jpg"
)

# Stylize using a named WikiArt style category
result = stylize(
    content_image="c:/art/val2017/000000000139.jpg",
    style="cubism",
    strength=0.50
)
```

### 2. Command-Line Inference

```bash
python inference/stylize.py --content c:/art/val2017/000000000139.jpg --style impressionism --strength 0.75
```

### 3. Interactive Simple UI

```bash
python -m streamlit run simple_ui.py
```

### 4. Training / Fine-tuning

```bash
# Verify pipeline on a tiny step run
python training/train.py --max_steps 10

# Full training run
python training/train.py --config configs/config.yaml
```
