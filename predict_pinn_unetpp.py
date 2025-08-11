import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

# Import your custom loss
from train_pinn_unetpp import physics_informed_loss

# Paths
model_path = "models/pinn_unetpp_best.keras"
test_dir = "data/test/images"
output_dir = "results/predictions"
os.makedirs(output_dir, exist_ok=True)

# Load model
model = load_model(model_path, custom_objects={'physics_informed_loss': physics_informed_loss})
print("✅ Model loaded successfully!")

# Loop through test images
for img_name in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_name)

    # Read image (grayscale for segmentation)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠ Could not read {img_path}")
        continue

    # Resize & normalize
    img_resized = cv2.resize(img, (128, 128))
    img_input = img_resized.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_input, axis=(0, -1))  # shape: (1, 128, 128, 1)

    # Prediction
    pred_mask = model.predict(img_input)[0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255  # binary mask

    # Save mask
    mask_save_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_mask.png")
    cv2.imwrite(mask_save_path, pred_mask)

    # Show side-by-side comparison
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(img_resized, cmap='gray')
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    axs[1].imshow(pred_mask, cmap='gray')
    axs[1].set_title("Predicted Mask")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

    print(f"✅ Saved: {mask_save_path}")

print("🎯 All predictions completed!")
