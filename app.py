import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Simulated model prediction output
def dummy_model_predict(image):
    # Pretend these are model's outputs
    return {
        "Pneumonia": 0.477,
        "Effusion": 0.463,
        "Infiltration": 0.442,
        "No Finding": 0.442
    }

# Load and display image
def load_image(uploaded_file):
    image = Image.open(uploaded_file).convert('RGB')
    return image

def main():
    st.set_page_config(page_title="Chest X-ray Diagnosis", layout="wide")

    st.title("💀 Upload & Prediction Settings")
    st.write("Upload a chest X-ray image. The model predicts possible findings and shows an explainable heatmap.")

    uploaded_file = st.file_uploader("Upload Chest X-ray", type=["png", "jpg", "jpeg"])

    threshold = st.slider('Select Confidence Threshold (lower = more sensitive)', 0.0, 1.0, 0.4, step=0.01)

    if uploaded_file is not None:
        # Load and display the uploaded image
        image = load_image(uploaded_file)
        st.image(image, caption="Uploaded Chest X-ray", width=400)

        # Get dummy model predictions
        outputs = dummy_model_predict(image)

        # Display Raw Model Outputs
        st.subheader("🔎 Raw Model Outputs:")
        st.json(outputs)

        # Apply threshold
        pred_labels = [label for label, prob in outputs.items() if prob >= threshold]

        if pred_labels:
            st.success(f"🩺 Predicted Findings: {', '.join(pred_labels)}")
        else:
            st.warning("⚠️ No findings above the selected threshold. Try lowering it.")

        # (Optional) Dummy GradCAM output
        st.subheader("🧠 Grad-CAM Heatmap (Simulated)")
        # Here you would generate GradCAM — I'm showing dummy
        fig, ax = plt.subplots()
        ax.imshow(np.random.rand(224, 224), cmap='jet')
        ax.axis('off')
        st.pyplot(fig)

if __name__ == "__main__":
    main()
