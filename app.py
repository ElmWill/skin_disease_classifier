import streamlit as st

st.set_page_config(page_title='Skin Disease Classifier', layout='centered')

st.title("🩺 Skin Disease Classifier")
st.write("""
This app classifies dermatoscopic skin lesion images into disease categories
using a transfer-learned EfficientNetB0 model. Use the pages in the left sidebar:
- Predict — upload an image and get a prediction + Grad-CAM overlay
- Explain Model — model summary and details
- About Project — dataset, methods, disclaimers
""")

st.info("⚠️ This is a demo research tool — **not for medical diagnosis**.")