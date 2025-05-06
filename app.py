import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from grad_cam import generate_grad_cam

# Set page
st.set_page_config(page_title="Medical Diagnosis", layout="wide")

# Load the model (cached)
@st.cache_resource
def load_trained_model():
    return load_model("compressed_model.h5")

model = load_trained_model()

# Define class names
class_names = ['Pneumonia', 'Effusion', 'Infiltration', 'No Finding']

# Sidebar
st.sidebar.title("🩻 Upload & Prediction Settings")
st.sidebar.markdown("Upload a chest X-ray image for diagnosis with explainable AI heatmap.")

# Main Title
st.title("🩺 Medical Diagnosis with Explainable AI")

# File uploader
uploaded_file = st.file_uploader("📤 Upload a Chest X-Ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Process the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

    img_resized = cv2.resize(img, (224, 224))
    img_normalized = img_resized / 255.0
    img_array = np.expand_dims(img_normalized, axis=0)

    with st.spinner("🧠 Analyzing Image..."):
        # Get predictions
        preds = model.predict(img_array, verbose=0)[0]
        
        # Get top prediction
        top_pred_index = np.argmax(preds)
        top_pred_class = class_names[top_pred_index]
        top_pred_prob = float(preds[top_pred_index])

        # Display raw outputs
        st.write("🔎 Model Output Probabilities:", {
            class_names[i]: float(preds[i]) 
            for i in range(len(class_names))
        })

        # Display primary prediction
        st.subheader("🔍 Diagnosis Prediction:")
        st.markdown(f"**{top_pred_class}** (confidence: {top_pred_prob:.2f})")

        # Generate and show Grad-CAM heatmap
        try:
            gradcam_image = generate_grad_cam(model, img_normalized, preds, class_names)
            st.subheader("🧠 Explainable AI Heatmap")
            st.caption(f"Showing important regions for: {top_pred_class}")
            st.image(gradcam_image, use_column_width=True)
        except Exception as e:
            st.error(f"Could not generate heatmap: {str(e)}")

else:
    st.markdown("Please upload a chest X-ray image to get started.")
