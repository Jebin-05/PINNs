import os
import numpy as np
import cv2
import random
import shutil

def ensure_val_split(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir, split_ratio=0.1):
    """
    Creates a validation set if it does not exist by splitting from train set.
    """
    if not os.path.exists(val_img_dir):
        os.makedirs(val_img_dir)
    if not os.path.exists(val_mask_dir):
        os.makedirs(val_mask_dir)

    if len(os.listdir(val_img_dir)) == 0:  # Only split if empty
        img_files = sorted(os.listdir(train_img_dir))
        mask_files = sorted(os.listdir(train_mask_dir))

        val_count = int(len(img_files) * split_ratio)
        selected_indices = random.sample(range(len(img_files)), val_count)

        for idx in selected_indices:
            shutil.move(os.path.join(train_img_dir, img_files[idx]), val_img_dir)
            shutil.move(os.path.join(train_mask_dir, mask_files[idx]), val_mask_dir)

        print(f"✅ Created validation set with {val_count} samples.")


def load_data(img_dir, mask_dir, img_size=(128, 128)):
    """
    Loads images and masks from the given directories, resizes them,
    and normalizes pixel values.
    """
    images, masks = [], []

    img_files = sorted(os.listdir(img_dir))
    mask_files = sorted(os.listdir(mask_dir))

    for img_file, mask_file in zip(img_files, mask_files):
        img_path = os.path.join(img_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            continue

        img = cv2.resize(img, img_size)
        mask = cv2.resize(mask, img_size)

        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        images.append(img)
        masks.append(mask)

    images = np.expand_dims(np.array(images), axis=-1)
    masks = np.expand_dims(np.array(masks), axis=-1)

    return images, masks
