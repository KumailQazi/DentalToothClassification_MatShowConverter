"""Streamlit demo: upload a dental X-ray and predict old vs teen."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import tensorflow as tf
from PIL import Image

from Convo2dModel import MODEL_PATH, predict_label

_PROJECT_ROOT = Path(__file__).resolve().parent


@st.cache_resource
def load_model() -> tf.keras.Model:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Train first: python Convo2dModel.py"
        )
    return tf.keras.models.load_model(MODEL_PATH)


def main() -> None:
    st.set_page_config(page_title="Dental X-ray Classifier", layout="centered")
    st.title("Dental X-ray: old vs teen")
    st.caption(
        "Research demo only — not for clinical use. "
        "Upload a cropped dental X-ray image (JPG/PNG)."
    )

    try:
        model = load_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    uploaded = st.file_uploader("X-ray image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Upload an image to run inference.")
        return

    pil_image = Image.open(uploaded)
    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_image, caption="Uploaded image", use_container_width=True)
    with col2:
        label, confidence, prob_teen, prob_old = predict_label(model, pil_image)
        st.metric("Prediction", label.upper())
        st.metric("Confidence", f"{confidence * 100:.1f}%")
        st.progress(min(max(confidence, 0.0), 1.0))
        st.caption(f"P(teen) = {prob_teen:.3f} · P(old) = {prob_old:.3f}")


if __name__ == "__main__":
    main()
