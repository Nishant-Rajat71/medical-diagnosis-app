import tensorflow as tf
import numpy as np
import cv2
import io
from PIL import Image

def generate_grad_cam(model, img_resized, preds, class_names):
    img_array = np.expand_dims(img_resized, axis=0)
    class_idx = np.argmax(preds)
    
    last_conv_layer_name = [layer.name for layer in model.layers if 'conv' in layer.name][-1]
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(conv_outputs * pooled_grads, axis=-1)

    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = cv2.resize(heatmap.numpy(), (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(np.uint8(img_resized * 255), 0.6, heatmap_color, 0.4, 0)
    _, buffer = cv2.imencode(".png", superimposed)
    return Image.open(io.BytesIO(buffer))
