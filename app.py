import streamlit as st

# ✅ Must be first Streamlit command
st.set_page_config(page_title="Medical Diagnosis", layout="wide")

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from grad_cam import generate_grad_cam  # Ensure grad_cam.py is in the same folder

# Load the compressed model
@st.cache_resource
def load_trained_model():
    return load_model("compressed_model.h5")  # ✅ Correct filename

model = load_trained_model()

# Class names in order your model was trained
class_names = ['Pneumonia', 'Effusion', 'Infiltration', 'No Finding']

st.title("🩺 Medical Diagnosis with Explainable AI")

uploaded_file = st.file_uploader("📤 Upload a Chest X-Ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Convert uploaded image to NumPy array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (224, 224)) / 255.0
    img_array = np.expand_dims(img_resized, axis=0)

    # Predict
    preds = model.predict(img_array)[0]
    pred_labels = [class_names[i] for i, val in enumerate(preds > 0.5) if val]

    st.subheader("🔍 Predicted Findings:")
    st.markdown(", ".join(pred_labels) if pred_labels else "**No Finding**")

    # Grad-CAM
    gradcam_image = generate_grad_cam(model, img_resized, preds, class_names)
    st.subheader("🧠 Grad-CAM Heatmap")
    st.image(gradcam_image, caption="Important Regions Highlighted", use_column_width=True)
