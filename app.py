import streamlit as st

# ✅ Must be first Streamlit command
st.set_page_config(page_title="Medical Diagnosis", layout="wide")

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from grad_cam import generate_grad_cam

# Load the compressed model
@st.cache_resource
def load_trained_model():
    return load_model("compressed_model.h5")  # ✅ Correct filename

model = load_trained_model()

# Class names in order your model was trained
class_names = ['Pneumonia', 'Effusion', 'Infiltration', 'No Finding']

# Sidebar for model info
st.sidebar.title("🩻 Upload & Prediction Settings")
# st.sidebar.markdown("### Model: MobileNetV2 (compressed) + Grad-CAM")
st.sidebar.markdown("This app uses a deep learning model to diagnose medical conditions from chest X-ray images.")
st.sidebar.markdown("**Instructions:** Upload a chest X-ray image, and the model will predict the possible findings with an explanation using Grad-CAM.")
st.sidebar.markdown("---")

# Allow user to adjust prediction confidence threshold
confidence_threshold = st.sidebar.slider("Select Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# Main title
st.title("🩺 Medical Diagnosis with Explainable AI")

# Image uploader
uploaded_file = st.file_uploader("📤 Upload a Chest X-Ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Convert uploaded image to NumPy array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (224, 224)) / 255.0
    img_array = np.expand_dims(img_resized, axis=0)

    # Show progress spinner while processing
    with st.spinner("🧠 Analyzing Image..."):
        # Predict
        preds = model.predict(img_array)[0]
        pred_labels = [class_names[i] for i, val in enumerate(preds > confidence_threshold) if val]

        # Display prediction results
        st.subheader("🔍 Predicted Findings:")
        if pred_labels:
            st.markdown(", ".join(pred_labels))
        else:
            st.markdown("**No Finding**")

        # Grad-CAM
        gradcam_image = generate_grad_cam(model, img_resized, preds, class_names)
        st.subheader("🧠 Grad-CAM Heatmap")
        st.image(gradcam_image, caption="Important Regions Highlighted", use_column_width=True)

else:
    st.markdown("👈 Please upload a chest X-ray image from the sidebar to get started.")
