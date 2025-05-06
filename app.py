import streamlit as st

# ✅ Set page
st.set_page_config(page_title="Medical Diagnosis", layout="wide")

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from grad_cam import generate_grad_cam

# ✅ Load the model (cached)
@st.cache_resource
def load_trained_model():
    return load_model("compressed_model.h5")

model = load_trained_model()

# ✅ Define class names
class_names = ['Pneumonia', 'Effusion', 'Infiltration', 'No Finding']

# ✅ Sidebar
st.sidebar.title("🩻 Upload & Prediction Settings")
st.sidebar.markdown("Upload a chest X-ray image. The model predicts possible findings and shows an explainable heatmap.")
st.sidebar.markdown("---")

# ✅ Confidence Threshold
confidence_threshold = st.sidebar.slider(
    "Select Confidence Threshold (lower = more sensitive)", 0.0, 1.0, 0.2, 0.05
)

# ✅ Main Title
st.title("🩺 Medical Diagnosis with Explainable AI")

# ✅ File uploader
uploaded_file = st.file_uploader("📤 Upload a Chest X-Ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # ✅ First, display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # ✅ Read and process the uploaded image (only once)
    file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    img_resized = cv2.resize(img, (224, 224))
    img_normalized = img_resized / 255.0  # Normalize
    img_array = np.expand_dims(img_normalized, axis=0)

    # ✅ Spinner
    with st.spinner("🧠 Analyzing Image..."):
        # ✅ Predict
        preds = model.predict(img_array)[0]

        # ✅ Safe conversion to dictionary
        raw_outputs = {class_names[i]: float(preds[i]) for i in range(len(class_names))}
        st.write("🔎 Raw Model Outputs:", raw_outputs)

        # ✅ Thresholding
        pred_labels = []
        for i, prob in enumerate(preds):
            if prob >= confidence_threshold:
                pred_labels.append(class_names[i])

        # ✅ Display prediction
        st.subheader("🔍 Predicted Findings:")
        if pred_labels:
            st.markdown(", ".join(pred_labels))
        else:
            st.markdown("**No strong findings detected.** (Try lowering the threshold)")

        # ✅ Grad-CAM
        gradcam_image = generate_grad_cam(model, img_normalized, preds, class_names)
        st.subheader("🧠 Grad-CAM Heatmap")
        st.image(gradcam_image, caption="Important Regions Highlighted", use_column_width=True)

else:
    st.markdown("Please upload a chest X-ray image from the sidebar to get started.")
