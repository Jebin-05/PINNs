import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
from U_Netpp import build_unetpp

# Paths
IMG_DIR = "data/train/images"
MASK_DIR = "data/train/masks"
MODEL_OUT = "models/unetpp_classifier.h5"
IMG_SIZE = (128, 128)

# Classes
class_names = ["Low salt zone", "Medium salt zone", "High salt zone"]

# Helper: Calculate salt percentage and assign class

def get_salt_class(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    salt_pct = np.sum(mask > 127) / mask.size * 100
    if salt_pct >= 30:
        return 2  # High
    elif salt_pct >= 10:
        return 1  # Medium
    else:
        return 0  # Low

# Gather image/mask pairs
img_files = sorted(os.listdir(IMG_DIR))
mask_files = sorted(os.listdir(MASK_DIR))
img_paths, labels = [], []
for fname in img_files:
    img_path = os.path.join(IMG_DIR, fname)
    mask_path = os.path.join(MASK_DIR, fname)
    if os.path.exists(mask_path):
        label = get_salt_class(mask_path)
        if label is not None:
            img_paths.append(img_path)
            labels.append(label)

# Split
train_imgs, val_imgs, train_labels, val_labels = train_test_split(img_paths, labels, test_size=0.1, random_state=42, stratify=labels)

# Data generator
class DataGen(tf.keras.utils.Sequence):
    def __init__(self, img_paths, labels, batch_size=32, shuffle=True):
        self.img_paths = img_paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(img_paths))
        self.on_epoch_end()
    def __len__(self):
        return int(np.ceil(len(self.img_paths) / self.batch_size))
    def __getitem__(self, idx):
        batch_idx = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_imgs = []
        batch_labels = []
        for i in batch_idx:
            img = cv2.imread(self.img_paths[i], cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, IMG_SIZE)
            img = np.expand_dims(img, axis=-1) / 255.0  # Single channel for UNet++
            batch_imgs.append(img)
            batch_labels.append(self.labels[i])
        return np.array(batch_imgs), tf.keras.utils.to_categorical(batch_labels, num_classes=3)
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# Model - Use UNet++ as feature extractor
unetpp = build_unetpp(input_shape=(128, 128, 1), num_classes=1)
# Remove the final segmentation layer and add classification head
x = unetpp.layers[-2].output  # Get features before final conv layer
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dense(64, activation="relu")(x)
out = Dense(3, activation="softmax")(x)
model = Model(unetpp.input, out)

# Freeze early layers, fine-tune later layers
for i, layer in enumerate(model.layers):
    if i < len(model.layers) // 2:
        layer.trainable = False
    else:
        layer.trainable = True

model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

# Train with more epochs and better monitoring
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
train_gen = DataGen(train_imgs, train_labels, batch_size=16)  # Smaller batch size
val_gen = DataGen(val_imgs, val_labels, batch_size=16, shuffle=False)

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-7)
]

model.fit(train_gen, validation_data=val_gen, epochs=25, callbacks=callbacks)
model.save(MODEL_OUT)
print(f"✅ Saved UNet++ classifier to {MODEL_OUT}")
