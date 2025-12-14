import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

MODEL_PATH = "models/trained_model.h5"

def predict_image(image_path):
    model = tf.keras.models.load_model(MODEL_PATH)

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    resize = tf.image.resize(img_rgb, (256, 256))
    prediction = model.predict(
        np.expand_dims(resize / 255, axis=0)
    )

    plt.imshow(resize.numpy().astype(int))
    plt.axis('off')
    plt.show()

    if prediction > 0.015:
        print("Predicted: Not Autistic")
    else:
        print("Predicted: Autistic")

# Example usage
# predict_image("data/test/Autistic.134.jpg")
