import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io

def generate_grad_cam(model, img_resized, preds, class_names):
    img_array = np.expand_dims(img_resized, axis=0)
    class_idx = np.argmax(preds)

    # Find the last convolutional layer automatically
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break

    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    # Pool the gradients over the width and height dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel by the corresponding pooled gradient
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    # Resize heatmap to original image size
    heatmap = cv2.resize(heatmap, (img_resized.shape[1], img_resized.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    # Apply colormap
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose the heatmap onto the image
    img_rgb = np.uint8(img_resized * 255)
    superimposed_img = cv2.addWeighted(img_rgb, 0.6, heatmap_color, 0.4, 0)

    _, buffer = cv2.imencode(".png", superimposed_img)
    return Image.open(io.BytesIO(buffer))
