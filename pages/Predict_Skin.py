import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom
import matplotlib.pyplot as plt
import os
import cv2 as cv

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # go up from /pages
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model_tf.keras")

CUSTOM_OBJECTS = {
    "EfficientNetB0": EfficientNetB0,
    "RandomFlip": RandomFlip,
    "RandomRotation": RandomRotation,
    "RandomZoom": RandomZoom,
}
st.write(f"Resolved Path: {MODEL_PATH}")
st.write(f"File Exists: {os.path.exists(MODEL_PATH)}")

CLASS_NAMES = ["nv", "mel", "bcc", "bkl"]

# -----------------------------
# Utilities
# -----------------------------
@st.cache_resource
def load_trained_model(path=MODEL_PATH):
    """
    Load the trained Keras model once and cache it.
    Using @st.cache_resource avoids reloading the model on every UI interaction.
    """
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=CUSTOM_OBJECTS)
    return model

def preprocess_img_pil(pil_img, target_size=(224,224)):
    """
    Convert a PIL image to a preprocessed numpy array ready for EfficientNet:
    - resize
    - convert to array
    - expand dims
    - apply EfficientNet preprocess_input (scaling, normalization)
    Returns array shape (1, H, W, 3)
    """
    img = pil_img.resize(target_size)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr

def make_gradcam_heatmap_tf(img_array, model, last_conv_layer_name="top_conv", pred_index=None):
    """
    Generate Grad-CAM heatmap (TensorFlow/Keras):
    - img_array: preprocessed array shape (1,H,W,3)
    - model: full keras model
    - last_conv_layer_name: name of the last conv layer in EfficientNetB0 (usually 'top_conv')
    Returns heatmap resized to conv feature map size (2D numpy array normalized 0..1)
    """
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        return np.zeros_like(heatmap.numpy())
    heatmap /= max_val
    return heatmap.numpy()

def overlay_heatmap_on_image(pil_img, heatmap, alpha=0.4, cmap="jet"):
    """
    Create overlay image (numpy uint8) showing heatmap on top of the original image.
    - pil_img: original PIL image (any size)
    - heatmap: 2D numpy array normalized 0..1 (must be resized to image size inside)
    """
    img = np.array(pil_img.convert("RGB"))
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv.applyColorMap(heatmap_uint8, cv.COLORMAP_JET)  # BGR
    heatmap_color = cv.cvtColor(heatmap_color, cv.COLOR_BGR2RGB)
    heatmap_resized = tf.image.resize(heatmap_color, (img.shape[0], img.shape[1])).numpy().astype(np.uint8)
    overlay = cv.addWeighted(img, 1 - alpha, heatmap_resized, alpha, 0)
    return img, heatmap_resized, overlay

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Predict — Skin Lesion Classification")
st.write("Upload a dermatoscopic image (jpg/png). The model will predict the class and show Grad-CAM.")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
model = load_trained_model()

if uploaded is not None:
    pil_img = image.load_img(uploaded)
    st.image(pil_img, caption="Uploaded image", use_column_width=True)

    # Preprocess and predict
    input_arr = preprocess_img_pil(pil_img, target_size=(224,224))
    preds = model.predict(input_arr)
    prob = np.max(preds)
    pred_idx = int(np.argmax(preds))
    pred_label = CLASS_NAMES[pred_idx]

    st.markdown(f"**Prediction:** `{pred_label}`  — **Confidence:** {prob*100:.2f}%")

    # Generate Grad-CAM heatmap for the predicted class
    try:
        heatmap = make_gradcam_heatmap_tf(input_arr, model, last_conv_layer_name="top_conv", pred_index=pred_idx)
        orig, heat_col, overlay = overlay_heatmap_on_image(pil_img, heatmap, alpha=0.4)

        # Display original, heatmap, overlay side by side
        st.write("Grad-CAM visualization:")
        cols = st.columns(3)
        cols[0].image(orig, caption="Original", use_column_width=True)
        cols[1].image(heat_col, caption="Heatmap", use_column_width=True)
        cols[2].image(overlay, caption="Overlay", use_column_width=True)
    except Exception as e:
        st.error(f"Grad-CAM failed: {e}")