import os
import cv2
from PIL import Image
import tensorflow as tf

IMAGE_EXTS = ['jpeg', 'jpg', 'bmp', 'png']

def clean_dataset(data_dir):
    for image_class in os.listdir(data_dir):
        class_path = os.path.join(data_dir, image_class)

        if not os.path.isdir(class_path):
            continue

        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            try:
                with Image.open(image_path) as img:
                    img.verify()  # verifies image integrity
            except Exception:
                print(f"Removing corrupt image: {image_path}")
                os.remove(image_path)

def load_dataset(data_dir, img_size=(256, 256)):
    data = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=img_size
    )
    data = data.map(lambda x, y: (x / 255, y))
    return data

def split_dataset(data, train_ratio=0.75, val_ratio=0.15):
    total = len(data)
    train_size = int(total * train_ratio) + 1
    val_size = int(total * val_ratio)

    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size + val_size)

    return train, val, test
