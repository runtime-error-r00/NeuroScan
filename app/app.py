"""NeuroScan: brain MRI tumor classification demo with Grad-CAM."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from matplotlib import colormaps
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import crop_black_borders, get_transforms
from src.gradcam import GradCAM
from src.model import load_checkpoint

CKPT_PATH = ROOT / "models" / "resnet18_best.pt"


@st.cache_resource
def load_model():
    """Loaded once and shared across reruns."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(CKPT_PATH, device)
    return model, ckpt, GradCAM(model, model.layer4)

def overlay(img, cam, alpha=0.4):
    """Blend a [0, 1] heatmap (jet colormap) over an RGB PIL image of the same size."""
    base = np.asarray(img, dtype=np.float32) / 255
    heat = colormaps["jet"](cam)[..., :3]
    return (1 - alpha) * base + alpha * heat


def analyze(img, model, ckpt, gradcam):
    """Same preprocessing as evaluation; returns the image the model sees, heatmap, class and probabilities."""
    size = ckpt["img_size"]
    x = get_transforms(train=False, img_size=size)(img)
    cam, cls, probs = gradcam(x)
    seen = crop_black_borders(img).resize((size, size))
    return seen, cam, cls, probs

st.set_page_config(page_title="NeuroScan", layout="wide")
st.title("NeuroScan")
st.caption("Brain MRI tumor classification (glioma, meningioma, pituitary, no tumor) with Grad-CAM")
st.warning("Educational project. **Not a medical device**: do not use for diagnosis.")

model, ckpt, gradcam = load_model()
classes = ckpt["classes"]

with st.sidebar:
    st.header("Model")
    st.write(f"`{ckpt['model_name']}`, epoch {ckpt['epoch']}")
    st.write(f"Validation macro-F1: **{ckpt['val_metrics']['macro_f1']:.3f}**")
    st.write("Test macro-F1: **0.978** (95% CI 0.968–0.987)")
    st.header("Limitations")
    st.markdown(
        "- Trained on one public Kaggle dataset (single 2D slices, no patient IDs).\n"
        "- Confidence is often high even when wrong.\n"
        "- *notumor* may partly rely on image style (MRI sequence) rather than anatomy.\n"
        "- Errors concentrate on gliomas with doubtful labels."
    )

uploaded = st.file_uploader("Upload a brain MRI slice", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload an image to get a prediction.")
    st.stop()

img = Image.open(uploaded).convert("RGB")
seen, cam, cls, probs = analyze(img, model, ckpt, gradcam)

left, right = st.columns(2)
left.image(seen, caption="Model input (cropped, 224×224)", width="stretch")
right.image(overlay(seen, cam), caption="Grad-CAM: regions driving the prediction",
            width="stretch", clamp=True)

st.subheader(f"Prediction: {classes[cls]} ({probs[cls]:.1%})")
st.bar_chart(pd.DataFrame({"probability": probs}, index=classes))
st.caption("Probabilities come from the model's softmax and are not calibrated: treat them as scores, not certainties.")