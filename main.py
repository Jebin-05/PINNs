# main.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model

from utils import load_data, build_unetpp, dice_coef, iou_metric, PhysicsInformedLoss

# ----------------- Directories -----------------
def create_project_directories():
    for d in ['models', 'results/predictions']:
        os.makedirs(d, exist_ok=True)

if __name__ == '__main__':
    create_project_directories()

    data_path = 'data/'
    train_images_path = os.path.join(data_path, 'train/images/')
    train_masks_path = os.path.join(data_path, 'train/masks/')

    # Load CSV and data
    train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
    image_ids = train_df['id'].values
    X, y = load_data(image_ids, train_images_path, train_masks_path, img_size=(256, 256))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    # Build & compile model
    loss_fn = PhysicsInformedLoss(w_data=1.0, w_physics=2.0)  # Increased w_physics for stronger PINN
    model = build_unetpp(input_shape=(256, 256, 1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=loss_fn,
        metrics=[dice_coef, iou_metric]
    )
    model.summary()

    # Callbacks
    checkpoint = ModelCheckpoint('models/best_model.h5', monitor='val_dice_coef',
                                  mode='max', save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_dice_coef', mode='max',
                               patience=15, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_dice_coef', mode='max',
                                  factor=0.5, patience=7, verbose=1)

    # Train (CPU-friendly)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=80,                # Reduced because U-Net++ is powerful
        batch_size=4,              # Small batch size for CPU
        callbacks=[checkpoint, early_stop, reduce_lr]
    )

    # Save training history
    pd.DataFrame(history.history).to_csv('results/training_history.csv', index=False)

    # Load best model
    best_model = load_model(
        'models/best_model.h5',
        custom_objects={'PhysicsInformedLoss': loss_fn,
                        'dice_coef': dice_coef, 'iou_metric': iou_metric}
    )

    # Predictions on validation
    preds = best_model.predict(X_val)
    preds_bin = (preds > 0.5).astype(np.float32)

    # Save some predictions
    for i in range(5):
        plt.imsave(f'results/predictions/val_{i}_pred.png', preds_bin[i].squeeze(), cmap='gray')
        plt.imsave(f'results/predictions/val_{i}_true.png', y_val[i].squeeze(), cmap='gray')
