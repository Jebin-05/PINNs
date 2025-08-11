import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from train_pinn_unetpp import physics_informed_loss

# --------------------
# CONFIG
# --------------------
MODEL_PATH = "models/pinn_unetpp_best.keras"
TEST_IMAGES_DIR = "data/val/images"
TEST_MASKS_DIR = "data/val/masks"
IMG_SIZE = (128, 128)

# --------------------
# Load Model
# --------------------
print("Loading model...")
model = load_model(MODEL_PATH, custom_objects={'physics_informed_loss': physics_informed_loss})
print("✅ Model loaded successfully!")

# --------------------
# Metrics
# --------------------
def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-7)

def iou_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / (union + 1e-7)

def pixel_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# --------------------
# Evaluation Loop
# --------------------
dice_scores = []
iou_scores = []
pixel_accs = []

image_files = sorted(os.listdir(TEST_IMAGES_DIR))

for filename in image_files:
    img_path = os.path.join(TEST_IMAGES_DIR, filename)
    mask_path = os.path.join(TEST_MASKS_DIR, filename)  # same name as image

    if not os.path.exists(mask_path):
        print(f"⚠ No mask found for {filename}, skipping...")
        continue

    # Load and preprocess
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    img_resized = cv2.resize(img, IMG_SIZE) / 255.0
    mask_resized = cv2.resize(mask, IMG_SIZE) / 255.0
    mask_resized = (mask_resized > 0.5).astype(np.uint8)

    input_img = np.expand_dims(img_resized, axis=(0, -1))

    # Prediction
    pred_mask = model.predict(input_img, verbose=0)[0, :, :, 0]
    pred_mask_bin = (pred_mask > 0.5).astype(np.uint8)

    # Metrics
    dice_scores.append(dice_coefficient(mask_resized, pred_mask_bin))
    iou_scores.append(iou_score(mask_resized, pred_mask_bin))
    pixel_accs.append(pixel_accuracy(mask_resized, pred_mask_bin))

# --------------------
# Results
# --------------------
print("\n📊 Evaluation Results:")
print(f"Dice Coefficient: {np.mean(dice_scores):.4f}")
print(f"IoU Score:        {np.mean(iou_scores):.4f}")
print(f"Pixel Accuracy:   {np.mean(pixel_accs):.4f}")
