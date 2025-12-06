import streamlit as st

st.title("About Project")
st.write("""
**Skin Disease Classifier** using HAM10000 dataset (subset: nv, mel, bcc, bkl).
This project demonstrates:
- Transfer learning with EfficientNetB0
- Class imbalance handling (class weights + augmentation)
- Explainable AI with Grad-CAM
- Deployment via Streamlit multipage app
""")

st.markdown("### Dataset")
st.write("""
- HAM10000: dermatoscopic images with 7 classes. For this project we used 4 common classes:
  - `nv` (melanocytic nevi)
  - `mel` (melanoma)
  - `bcc` (basal cell carcinoma)
  - `bkl` (benign keratosis-like)
- Split: 80% train / 10% val / 10% test (stratified)
""")

st.markdown("### Important disclaimer")
st.warning("""
This app is for **research / educational** purposes only.
It is **not a medical diagnosis tool**. Always consult a licensed dermatologist/clinician for medical advice.
""")
