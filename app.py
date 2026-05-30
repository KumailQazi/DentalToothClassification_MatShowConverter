"""Streamlit demo: upload a dental X-ray and predict old vs teen."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from Convo2dModel import CLASS_INDICES_PATH, IMAGE_SIZE, MODEL_PATH

_PROJECT_ROOT = Path(__file__).resolve().parent


@st.cache_resource
def load_model() -> tf.keras.Model:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Train first: python Convo2dModel.py"
        )
    return tf.keras.models.load_model(MODEL_PATH)


def load_class_indices() -> dict[str, int]:
    if CLASS_INDICES_PATH.is_file():
        return json.loads(CLASS_INDICES_PATH.read_text())
    return {"old": 0, "teen": 1}


def predict_label(model: tf.keras.Model, pil_image: Image.Image) -> tuple[str, float, float]:
    """Return (label, confidence, P(teen))."""
    class_indices = load_class_indices()
    index_to_label = {int(v): k for k, v in class_indices.items()}
    teen_index = class_indices.get("teen", 1)

    img = pil_image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.asarray(img, dtype=np.float32)
    batch = np.expand_dims(arr, axis=0) / 255.0
    prob_teen = float(model.predict(batch, verbose=0)[0][0])
    pred_index = teen_index if prob_teen >= 0.5 else (1 - teen_index)
    label = index_to_label.get(pred_index, "unknown")
    confidence = prob_teen if pred_index == teen_index else 1.0 - prob_teen
    return label, confidence, prob_teen


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
        label, confidence, prob_teen = predict_label(model, pil_image)
        st.metric("Prediction", label.upper())
        st.metric("Confidence", f"{confidence * 100:.1f}%")
        st.progress(min(max(confidence, 0.0), 1.0))
        st.caption(f"P(teen) = {prob_teen:.3f}")


if __name__ == "__main__":
    main()
