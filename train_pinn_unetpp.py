import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from U_Netpp import build_unetpp
from utils import load_data

# -----------------------
# Physics-Informed Loss
# -----------------------
def physics_informed_loss(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

    # Approximate derivatives (simple finite difference)
    dx = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]
    dy = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]

    laplacian_x = dx[:, 1:, :, :] - dx[:, :-1, :, :]
    laplacian_y = dy[:, :, 1:, :] - dy[:, :, :-1, :]

    laplacian = laplacian_x[:, :-1, :, :] + laplacian_y[:, :, :-1, :]

    physics_loss = tf.reduce_mean(tf.square(laplacian))
    return mse_loss + 0.1 * physics_loss


# -----------------------
# Paths
# -----------------------
train_img_dir = "data/train/images"
train_mask_dir = "data/train/masks"
val_img_dir = "data/val/images"
val_mask_dir = "data/val/masks"

# -----------------------
# Main Training Execution
# -----------------------
if __name__ == "__main__":
    print("Loading data...")
    X_train, y_train = load_data(train_img_dir, train_mask_dir, (128, 128))
    X_val, y_val = load_data(val_img_dir, val_mask_dir, (128, 128))

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    print("Building UNet++ model...")
    model = build_unetpp(input_shape=(128, 128, 1))

    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss=physics_informed_loss,
                  metrics=["accuracy"])

    # Callbacks
    checkpoint_cb = ModelCheckpoint("models/pinn_unetpp_best.keras", save_best_only=True, monitor="val_loss", mode="min")
    early_stopping_cb = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    reduce_lr_cb = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=4,
        epochs=20,
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb]
    )

    model.save("models/pinn_unetpp_final.keras")
    print("✅ Model saved to models/pinn_unetpp_final.keras")
