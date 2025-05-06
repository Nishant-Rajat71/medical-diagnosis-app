import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io

def generate_grad_cam(model, img_normalized, preds, class_names):
    try:
        # Ensure proper input shape
        if len(img_normalized.shape) == 3:
            img_array = np.expand_dims(img_normalized, axis=0)
        else:
            img_array = img_normalized
            
        # Get the predicted class index
        class_idx = np.argmax(preds)
        
        # Find the last convolutional layer
        conv_layers = [layer.name for layer in model.layers 
                      if 'conv' in layer.name.lower() or 'final_conv' in layer.name.lower()]
        if not conv_layers:
            raise ValueError("No convolutional layers found in the model")
        last_conv_layer_name = conv_layers[-1]
        
        # Create gradient model
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )

        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            raise ValueError("Gradients could not be computed (None returned)")
            
        # Generate heatmap
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_mean(conv_outputs * pooled_grads, axis=-1)

        # Process heatmap
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)  # Normalize
        heatmap = cv2.resize(heatmap.numpy(), (img_normalized.shape[1], img_normalized.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Superimpose on original image
        img_uint8 = np.uint8(img_normalized * 255)
        superimposed = cv2.addWeighted(img_uint8, 0.6, heatmap_color, 0.4, 0)
        
        # Convert to PIL Image
        _, buffer = cv2.imencode(".png", cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
        return Image.open(io.BytesIO(buffer))
        
    except Exception as e:
        raise RuntimeError(f"Grad-CAM generation failed: {str(e)}")
