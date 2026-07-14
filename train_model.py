"""
Fire Detection System - Model Training Script
------------------------------------------------
This script trains a Convolutional Neural Network (CNN) to classify
images as "Fire" or "No Fire" using TensorFlow/Keras.

FOLDER STRUCTURE REQUIRED:
dataset/
    fire_images/        <- put all fire photos here
    non_fire_images/    <- put all non-fire photos here

HOW TO RUN:
    python train_model.py
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Some images (especially in datasets downloaded from the internet) are
# slightly truncated - they open fine at first but error out partway through
# decoding. This tells PIL to load what it can instead of crashing on them.
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ------------------------------
# 1. SETTINGS
# ------------------------------
IMG_SIZE = (224, 224)      # every image will be resized to this
BATCH_SIZE = 32
EPOCHS = 15
DATASET_DIR = "dataset"    # folder containing fire_images/ and non_fire_images/
MODEL_SAVE_PATH = "model/model.h5"

# ------------------------------
# 2. LOAD & PREPROCESS DATA
# ------------------------------
# ImageDataGenerator automatically:
# - reads images from folders
# - resizes them
# - normalizes pixel values (0-255 -> 0-1)
# - splits into training and validation sets
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,       # 80% train, 20% validation
    rotation_range=15,
    zoom_range=0.15,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

val_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

print("Class labels:", train_data.class_indices)
# Example output: {'fire_images': 0, 'non_fire_images': 1}

# ------------------------------
# 3. BUILD THE CNN MODEL
# ------------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")   # 1 output neuron: probability of "non-fire"
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ------------------------------
# 4. TRAIN THE MODEL
# ------------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ------------------------------
# 5. SAVE THE TRAINED MODEL
# ------------------------------
import os
os.makedirs("model", exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"\n Model saved successfully at: {MODEL_SAVE_PATH}")

# ------------------------------
# 6. PLOT TRAINING RESULTS (optional but useful for your report)
# ------------------------------
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.legend()

plt.tight_layout()
plt.savefig("model/training_graph.png")
print("Training graph saved at: model/training_graph.png")
