import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ---------------------------
# CONFIGURATION
# ---------------------------
SEGMENTATION_MODEL_PATH = "models/pinn_unetpp_best.keras"
CLASSIFIER_MODEL_PATH = "models/unetpp_classifier.h5"
TEST_IMAGE_PATH = "/home/jebin/Desktop/PINNs/data/test/images/fffd909d0f.png"
IMG_SIZE = (128, 128)
RESULTS_DIR = "results/explainability"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Import custom loss
from train_pinn_unetpp import physics_informed_loss

# ---------------------------
# Load Segmentation Model
# ---------------------------
print("🔄 Loading segmentation model...")
seg_model = load_model(
    SEGMENTATION_MODEL_PATH,
    custom_objects={'physics_informed_loss': physics_informed_loss}
)
print("✅ Segmentation model loaded!")

# ---------------------------
# Load Classification Model
# ---------------------------
print("🔄 Loading UNet++ classification model...")
clf_model = load_model(CLASSIFIER_MODEL_PATH)
print("✅ Classification model loaded!")

# ---------------------------
# Preprocess Image
# ---------------------------
if not os.path.exists(TEST_IMAGE_PATH):
    raise FileNotFoundError(f"❌ Test image not found: {TEST_IMAGE_PATH}")

img = cv2.imread(TEST_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError(f"❌ Failed to read image: {TEST_IMAGE_PATH}")

img_resized = cv2.resize(img, IMG_SIZE) / 255.0
input_img_seg = np.expand_dims(img_resized, axis=(0, -1))  # for UNet++
input_img_clf = np.expand_dims(img_resized, axis=(0, -1))  # UNet++ classifier also expects single channel

# ---------------------------
# Segmentation Inference
# ---------------------------
pred_mask = seg_model.predict(input_img_seg, verbose=0)[0, :, :, 0]
binary_mask = (pred_mask > 0.5).astype(np.uint8)
salt_percentage = (np.sum(binary_mask) / binary_mask.size) * 100

h, w = binary_mask.shape
vertical = "left" if np.sum(binary_mask[:, :w//2]) > np.sum(binary_mask[:, w//2:]) else "right"
horizontal = "upper" if np.sum(binary_mask[:h//2, :]) > np.sum(binary_mask[h//2:, :]) else "lower"
region_text = f"{horizontal} {vertical}"

# ---------------------------
# Classification Inference
# ---------------------------
pred_class = clf_model.predict(input_img_clf, verbose=0)
class_label = np.argmax(pred_class)
class_names = ["Low salt zone", "Medium salt zone", "High salt zone"]
classified_zone = class_names[class_label]

# ---------------------------
# Grad-CAM
# ---------------------------
def get_gradcam_heatmap(model, image_array, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        loss = predictions[:, :, :, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0] * pooled_grads
    heatmap = tf.reduce_mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
    return heatmap

last_conv_layer = next((layer.name for layer in reversed(seg_model.layers)
                        if isinstance(layer, tf.keras.layers.Conv2D)), None)
if not last_conv_layer:
    raise ValueError("❌ No Conv2D layer found for Grad-CAM.")

heatmap = get_gradcam_heatmap(seg_model, input_img_seg, last_conv_layer)
heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 0.6, heatmap_colored, 0.4, 0)

cv2.imwrite(os.path.join(RESULTS_DIR, "gradcam_overlay.png"), overlay)

# ---------------------------
# Analysis and Reporting
# ---------------------------

strength = (
    "strong" if salt_percentage >= 30 else
    "moderate" if salt_percentage >= 10 else
    "weak"
)
viability = (
    "Highly viable for extraction." if salt_percentage >= 40 else
    "Moderately viable with support." if salt_percentage >= 20 else
    "Not currently viable for extraction."
)
region_patterns = {
    "upper left": "continental shelf edges",
    "upper right": "folded tectonic margins",
    "lower left": "subduction zones",
    "lower right": "oceanic trench peripheries"
}
similar_region = region_patterns.get(region_text, "offshore marine basin")

prompt = (
    f"The segmentation model predicts a {strength} salt deposit with approximately {salt_percentage:.2f}% coverage. "
    f"The classified region is: {classified_zone}. The location of salt is concentrated in the {region_text}. "
    f"Viability for extraction: {viability} Similar geological regions include {similar_region}."
)

# Simple rule-based explanation instead of GPT-2
explanation = (
    f"This geological formation shows {strength} salt presence with {salt_percentage:.2f}% coverage in the {region_text} region. "
    f"Based on the classification as {classified_zone.lower()}, this formation is characteristic of {similar_region}. "
    f"{viability}"
)

# ---------------------------
# Final Output
# ---------------------------
print("\n🧠 Final Geological Summary:")
print(f"- Salt Strength: {strength.capitalize()}")
print(f"- Coverage: {salt_percentage:.2f}%")
print(f"- Region: {region_text}")
print(f"- Viability: {viability}")
print(f"- Classifier Zone: {classified_zone}")
print(f"- Similar Pattern: {similar_region}")
print(f"- AI Interpretation: {explanation}\n")

# ---------------------------
# Visual Output
# ---------------------------
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Predicted Salt Mask")
plt.imshow(binary_mask, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Grad-CAM Overlay")
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()
