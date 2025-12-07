import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.models import load_model
import streamlit as st
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model_tf.keras")

model = None   # 🔥 Prevent NameError

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("Model Loaded Successfully 🔥")
except Exception as e:
    st.error("❌ Failed to load the model")
    st.exception(e)

# Show summary only if model exists
if model:
    st.subheader("📌 Model Summary")
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    summary = "\n".join(stringlist)
    st.text(summary)
else:
    st.warning("Model could not be loaded — summary skipped.")

st.subheader("What each part does")
st.write("""
- **EfficientNetB0 backbone**: pre-trained feature extractor (learned edges, textures, patterns).
- **GlobalAveragePooling2D**: compresses spatial features into a vector.
- **Dense + Softmax head**: maps features to class probabilities.
- **Grad-CAM**: post-hoc explainability showing which pixels contributed to prediction.
""")

st.subheader("Training details (brief)")
st.write("""
- Transfer learning: frozen backbone then fine-tuned with a small learning rate.
- Loss: Categorical crossentropy (multi-class).
- Class-weighting applied to handle imbalance.
- EarlyStopping & ModelCheckpoint used to avoid overfitting and save best model.
""")
