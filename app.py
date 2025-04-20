import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils.grad_cam import generate_grad_cam

@st.cache_resource
def load_trained_model():
    return load_model("model/trained_model.h5")

model = load_trained_model()
class_names = ['Pneumonia', 'Effusion', 'Infiltration', 'No Finding']

st.title("🩺 Medical Diagnosis with Explainable AI")

uploaded_file = st.file_uploader("Upload Chest X-Ray", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_resized = cv2.resize(img, (224, 224)) / 255.0
    img_array = np.expand_dims(img_resized, axis=0)

    preds = model.predict(img_array)[0]
    pred_labels = [class_names[i] for i, val in enumerate(preds > 0.5) if val]

    st.subheader("🔍 Prediction")
    st.markdown(", ".join(pred_labels) if pred_labels else "No Finding")

    gradcam_image = generate_grad_cam(model, img_resized, preds, class_names)
    st.image(gradcam_image, caption="Grad-CAM Heatmap", use_column_width=True)