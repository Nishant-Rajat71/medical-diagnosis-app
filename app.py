import streamlit as st

# ✅ Must be first Streamlit command
st.set_page_config(page_title="Medical Diagnosis", layout="wide")

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from grad_cam import generate_grad_cam

# 🧠 Load the compressed model
@st.cache_resource
def load_trained_model():
    return load_model("compressed_model.h5")

model = load_trained_model()
class_names = ['Pneumonia', 'Effusion', 'Infiltration', 'No Finding']

# 🎨 Sidebar
st.sidebar.title("🩻 Upload Chest X-ray")
uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
st.sidebar.markdown("---")
st.sidebar.info("Model: MobileNetV2 + Grad-CAM\n\nSize: ~26MB\nLabels: 4")

st.title("🩺 AI Medical Diagnosis with Explainable Grad-CAM")

# 🖼️ Main content
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (224, 224)) / 255.0
    img_array = np.expand_dims(img_resized, axis=0)

    with st.spinner("Analyzing image..."):
        preds = model.predict(img_array)[0]
        pred_labels = [class_names[i] for i, val in enumerate(preds > 0.5) if val]
        gradcam_image = generate_grad_cam(model, img_resized, preds, class_names)

    # 📊 Show predictions with confidence
    st.subheader("🔍 Prediction Results")
    for i, prob in enumerate(preds):
        st.markdown(f"**{class_names[i]}**: {prob * 100:.2f}% {'✅' if prob > 0.5 else ''}")

    # 🖼️ Display images side by side
    st.subheader("📷 Visual Explanation")
    col1, col2 = st.columns(2)

    with col1:
        st.image(uploaded_file, caption="Original X-ray", width=300)
    with col2:
        st.image(gradcam_image, caption="Grad-CAM Heatmap", width=300)

else:
    st.markdown("👈 Please upload a chest X-ray image from the **sidebar** to get started.")
