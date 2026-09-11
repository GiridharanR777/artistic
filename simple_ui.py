"""
Ultra-Fast Interactive UI for Artistic Style Transfer using Streamlit.
Optimized for 2-core CPU performance:
- In-memory model caching via @st.cache_resource (zero disk reloads)
- Deep feature caching in session_state (strength slider updates run in ~2 seconds!)
- Thread optimization (torch.set_num_threads(2))
- Immediate image display with on-demand quality metrics
"""

import os
import time
import glob
from PIL import Image
import torch
import torchvision.transforms as transforms
import streamlit as st

# Optimize PyTorch CPU execution
torch.set_num_threads(2)

from paths import ensure_weights_exist, SAMPLES_DIR, COCO_DIR, WIKIART_CACHE_DIR
ensure_weights_exist()

from inference.stylize import get_model
from models.adain import adaptive_instance_normalization
from data.preprocessing import tensor_to_image, rgb_to_ycbcr, ycbcr_to_rgb
from evaluation.metrics import evaluate_transfer_pair


st.set_page_config(
    page_title="Artistic Image Style Transfer",
    page_icon="🎨",
    layout="wide"
)

# 1. In-Memory Cached Model (Loads once into RAM)
@st.cache_resource
def load_app_model():
    return get_model()


model = load_app_model()

st.title("🎨 Artistic Image Style Transfer")
st.caption("Fast local neural style transfer with controllable strength and strict content preservation.")

# Sidebar Controls
st.sidebar.header("⚙️ Configuration")

strength = st.sidebar.slider(
    "Style Strength (α)",
    min_value=0.0,
    max_value=1.0,
    value=0.60,
    step=0.05,
    help="0.0 = pure content, 1.0 = maximum style strokes."
)

color_preserve = st.sidebar.checkbox(
    "Preserve Content Colors",
    value=False,
    help="Applies artistic texture while keeping the original content colors."
)

res_choice = st.sidebar.select_slider(
    "Processing Resolution",
    options=[256, 384, 512],
    value=256,
    format_func=lambda x: f"{x} × {x} (Fast ~3–5s)" if x == 256 else (f"{x} × {x} (Balanced ~8s)" if x == 384 else f"{x} × {x} (High Res ~20s)"),
    help="256x256 is recommended for real-time responsiveness on CPU."
)

# Two-column layout for input selection
col1, col2 = st.columns(2)

# Content Image Input
with col1:
    st.subheader("1. Content Image")
    content_source = st.radio("Content Source", ["Sample COCO", "Upload Image"], horizontal=True)
    content_img = None

    if content_source == "Sample COCO":
        sample_files = []
        if os.path.exists(COCO_DIR):
            sample_files = sorted(glob.glob(os.path.join(COCO_DIR, "*.jpg")))[:30]
        if not sample_files:
            sample_files = sorted(glob.glob(os.path.join(SAMPLES_DIR, "content", "*.jpg")))

        if sample_files:
            selected_sample = st.selectbox("Choose Sample Image", sample_files, format_func=os.path.basename)
            content_img = Image.open(selected_sample).convert("RGB")
            st.image(content_img, caption="Content Image", use_container_width=True)
    else:
        uploaded_c = st.file_uploader("Upload Content Photo", type=["jpg", "jpeg", "png"])
        if uploaded_c:
            content_img = Image.open(uploaded_c).convert("RGB")
            st.image(content_img, caption="Uploaded Content", use_container_width=True)

# Style Image Input
with col2:
    st.subheader("2. Artistic Style")
    style_source = st.radio("Style Source", ["WikiArt Collection", "Upload Image"], horizontal=True)
    style_img = None

    if style_source == "WikiArt Collection":
        categories = []
        if os.path.exists(WIKIART_CACHE_DIR):
            categories = sorted([d for d in os.listdir(WIKIART_CACHE_DIR) if os.path.isdir(os.path.join(WIKIART_CACHE_DIR, d))])

        if categories:
            chosen_cat = st.selectbox("Art Movement", categories)
            cat_files = sorted(glob.glob(os.path.join(WIKIART_CACHE_DIR, chosen_cat, "*.*")))
            if cat_files:
                chosen_style_file = st.selectbox("Artwork", cat_files, format_func=os.path.basename)
                style_img = Image.open(chosen_style_file).convert("RGB")
                st.image(style_img, caption=f"Style: {chosen_cat}", use_container_width=True)
        else:
            sample_styles = sorted(glob.glob(os.path.join(SAMPLES_DIR, "styles", "*.*")))
            if sample_styles:
                chosen_style_file = st.selectbox("Artwork", sample_styles, format_func=os.path.basename)
                style_img = Image.open(chosen_style_file).convert("RGB")
                st.image(style_img, caption="Artwork Sample", use_container_width=True)
    else:
        uploaded_s = st.file_uploader("Upload Style Artwork", type=["jpg", "jpeg", "png"])
        if uploaded_s:
            style_img = Image.open(uploaded_s).convert("RGB")
            st.image(style_img, caption="Uploaded Artwork", use_container_width=True)

st.markdown("---")

# Session state initialization for fast slider caching
if "cached_content_id" not in st.session_state:
    st.session_state.cached_content_id = None
    st.session_state.cached_style_id = None
    st.session_state.fc = None
    st.session_state.fs = None
    st.session_state.last_result = None
    st.session_state.last_time = 0.0

# Stylization Action
if st.button("🚀 Apply Artistic Style Transfer", type="primary"):
    if content_img is None:
        st.error("Please provide or select a Content Image.")
    elif style_img is None:
        st.error("Please provide or select a Style Image.")
    else:
        t_start = time.time()
        with st.spinner(f"Generating artwork at {res_choice}×{res_choice} on local CPU..."):
            to_tensor = transforms.ToTensor()
            device = torch.device("cpu")

            # Check if we can reuse previously encoded features
            content_id = (id(content_img), res_choice, color_preserve)
            style_id = (id(style_img), res_choice, color_preserve)

            with torch.inference_mode():
                if color_preserve:
                    # Luminance style transfer
                    c_resized = content_img.resize((res_choice, res_choice))
                    s_resized = style_img.resize((res_choice, res_choice))
                    c_t = to_tensor(c_resized).unsqueeze(0).to(device)
                    s_t = to_tensor(s_resized).unsqueeze(0).to(device)

                    c_ycbcr = rgb_to_ycbcr(c_t)
                    s_ycbcr = rgb_to_ycbcr(s_t)
                    c_y = c_ycbcr[:, 0:1, :, :].repeat(1, 3, 1, 1)
                    s_y = s_ycbcr[:, 0:1, :, :].repeat(1, 3, 1, 1)

                    fc = model.encoder.encode(c_y)
                    fs = model.encoder.encode(s_y)
                    t_feat = strength * adaptive_instance_normalization(fc, fs) + (1.0 - strength) * fc
                    styled_y_rgb = model.decoder(t_feat)
                    styled_ycbcr = rgb_to_ycbcr(styled_y_rgb)
                    merged = torch.cat([styled_ycbcr[:, 0:1, :, :], c_ycbcr[:, 1:2, :, :], c_ycbcr[:, 2:3, :, :]], dim=1)
                    out_tensor = ycbcr_to_rgb(merged)
                else:
                    # Standard RGB transfer with feature reuse
                    if (st.session_state.cached_content_id != content_id or
                        st.session_state.cached_style_id != style_id or
                        st.session_state.fc is None):
                        c_resized = content_img.resize((res_choice, res_choice))
                        s_resized = style_img.resize((res_choice, res_choice))
                        c_t = to_tensor(c_resized).unsqueeze(0).to(device)
                        s_t = to_tensor(s_resized).unsqueeze(0).to(device)

                        st.session_state.fc = model.encoder.encode(c_t)
                        st.session_state.fs = model.encoder.encode(s_t)
                        st.session_state.cached_content_id = content_id
                        st.session_state.cached_style_id = style_id

                    fc = st.session_state.fc
                    fs = st.session_state.fs
                    t_feat = strength * adaptive_instance_normalization(fc, fs) + (1.0 - strength) * fc
                    out_tensor = model.decoder(t_feat)

                result_img = tensor_to_image(out_tensor)

        t_elapsed = round(time.time() - t_start, 2)
        st.session_state.last_result = result_img
        st.session_state.last_time = t_elapsed
        st.session_state.last_content = content_img
        st.session_state.last_style = style_img

# Display Results if available
if st.session_state.last_result is not None:
    st.success(f"⚡ Stylization complete in **{st.session_state.last_time} seconds**!")

    r_col1, r_col2, r_col3 = st.columns(3)
    with r_col1:
        st.image(st.session_state.last_content.resize((res_choice, res_choice)), caption="Original Content", use_container_width=True)
    with r_col2:
        st.image(st.session_state.last_style.resize((res_choice, res_choice)), caption="Target Style", use_container_width=True)
    with r_col3:
        mode_label = " (Color-Preserved)" if color_preserve else ""
        st.image(st.session_state.last_result, caption=f"Stylized Result (α = {strength}){mode_label}", use_container_width=True)

    # Download Button
    out_path = "c:/art/outputs/latest_styled.png"
    st.session_state.last_result.save(out_path)
    with open(out_path, "rb") as f:
        st.download_button(
            label="💾 Download Stylized Artwork",
            data=f,
            file_name="stylized_artwork.png",
            mime="image/png"
        )

    # Optional On-Demand Metrics (does not slow down main transfer)
    with st.expander("📊 Quality & Evaluation Metrics (Optional)"):
        if st.button("Compute Metrics for Current Result"):
            with st.spinner("Calculating SSIM, PSNR, and feature distances..."):
                metrics = evaluate_transfer_pair(
                    st.session_state.last_result,
                    st.session_state.last_content,
                    st.session_state.last_style,
                    model.encoder
                )
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("SSIM (Structure)", f"{metrics['SSIM (Content Preservation)']:.4f}")
            m2.metric("PSNR (Reconstruction)", f"{metrics['PSNR (Reconstruction dB)']:.2f} dB")
            m3.metric("Content Distance", f"{metrics['Content Feature Distance']:.4f}")
            m4.metric("Style Distance", f"{metrics['Style Feature Distance']:.4f}")
