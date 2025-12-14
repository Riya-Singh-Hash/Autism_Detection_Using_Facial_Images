import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

from utils import setup_gpu
from preprocessing import clean_dataset, load_dataset, split_dataset



DATA_DIR = "data/train"
MODEL_PATH = "models/trained_model.h5"

setup_gpu()
clean_dataset(DATA_DIR)

data = load_dataset(DATA_DIR)
train, val, test = split_dataset(data)

model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D(),
    BatchNormalization(),

    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(),
    BatchNormalization(),

    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D(),
    BatchNormalization(),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
]

history = model.fit(
    train,
    validation_data=val,
    epochs=50,
    callbacks=callbacks
)

# Evaluation
pre, re, acc = Precision(), Recall(), BinaryAccuracy()

for X, y in test:
    preds = model.predict(X)
    pre.update_state(y, preds)
    re.update_state(y, preds)
    acc.update_state(y, preds)

print("Precision:", pre.result().numpy())
print("Recall:", re.result().numpy())
print("Accuracy:", acc.result().numpy())

model.save(MODEL_PATH)
