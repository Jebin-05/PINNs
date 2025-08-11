import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from train_pinn_unetpp import physics_informed_loss

# -----------------------
# Load Best Model
# -----------------------
model_path = "models/pinn_unetpp_best.keras"

print("Loading model...")
model = load_model(model_path, custom_objects={'physics_informed_loss': physics_informed_loss})
print("✅ Model loaded successfully!")
model.summary()

# -----------------------
# Function to Predict on a Single Image
# -----------------------
def predict_single_image(image_path, target_size=(128, 128)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)   # Add batch dimension

    prediction = model.predict(img)
    prediction = (prediction[0, :, :, 0] > 0.5).astype(np.uint8)  # Binary mask
    return prediction

# -----------------------
# Example Usage
# -----------------------
test_image_path = "/home/jebin/Desktop/PINNs/data/test/images/00a6bfc7a7.png"  # Change to your image
if os.path.exists(test_image_path):
    pred_mask = predict_single_image(test_image_path)
    print("Prediction mask shape:", pred_mask.shape)

    # Save the mask
    save_path = "results/predictions/sample_mask.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, pred_mask * 255)
    print(f"✅ Prediction mask saved to {save_path}")
else:
    print(f"⚠ Test image not found: {test_image_path}")
