# check.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

from utils import load_data, physics_informed_loss, dice_coef, iou_metric

# ----------------- Paths -----------------
DATA_PATH = 'data/'
TRAIN_IMAGES_PATH = os.path.join(DATA_PATH, 'train/images/')
TRAIN_MASKS_PATH = os.path.join(DATA_PATH, 'train/masks/')
CSV_PATH = os.path.join(DATA_PATH, 'train.csv')
MODEL_PATH = 'models/best_model.h5'

# ----------------- Load Data -----------------
print("📥 Loading data...")
train_df = pd.read_csv(CSV_PATH)
image_ids = train_df['id'].values
X, y = load_data(image_ids, TRAIN_IMAGES_PATH, TRAIN_MASKS_PATH)

# Same split as in main.py
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# ----------------- Load Model -----------------
print("📦 Loading best model...")
model = load_model(MODEL_PATH,
                   custom_objects={
                       'physics_informed_loss': physics_informed_loss,
                       'dice_coef': dice_coef,
                       'iou_metric': iou_metric
                   })
print("✅ Model loaded successfully!")

# ----------------- Evaluate -----------------
print("📊 Evaluating on validation set...")
val_loss, val_dice, val_iou = model.evaluate(X_val, y_val, verbose=1)
print(f"\n📌 Validation Results:\n Loss = {val_loss:.4f}\n Dice = {val_dice:.4f}\n IoU  = {val_iou:.4f}")

# ----------------- Save Predictions -----------------
print("💾 Saving some predictions to results/predictions/")
os.makedirs('results/predictions', exist_ok=True)

preds = model.predict(X_val)
preds_bin = (preds > 0.5).astype(np.float32)

for i in range(5):
    plt.imsave(f'results/predictions/val_{i}_pred.png', preds_bin[i].squeeze(), cmap='gray')
    plt.imsave(f'results/predictions/val_{i}_true.png', y_val[i].squeeze(), cmap='gray')

print("✅ Done! Check results/predictions/ for output masks.")
