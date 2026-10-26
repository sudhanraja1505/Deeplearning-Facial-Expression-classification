import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_model.keras")

# FER-2013 dataset labels:
# 0 Angry, 1 Disgust, 2 Fear, 3 Happy, 4 Sad, 5 Surprise, 6 Neutral
CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def load_fer2013_from_directory():
    """
    Expected directory structure:
    dataset/fer2013/train/angry/*.jpg
    dataset/fer2013/train/disgust/*.jpg
    ...
    dataset/fer2013/test/angry/*.jpg
    ...
    """
    data_dir = os.path.join(BASE_DIR, "dataset", "fer2013")
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        raise FileNotFoundError(
            "FER-2013 dataset not found. Put the dataset in dataset/fer2013/train "
            "and dataset/fer2013/test with one subfolder per emotion. See README.md."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, labels="inferred", label_mode="int", color_mode="grayscale",
        image_size=(48, 48), batch_size=64, shuffle=True, seed=42
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, labels="inferred", label_mode="int", color_mode="grayscale",
        image_size=(48, 48), batch_size=64, shuffle=False
    )

    expected = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    actual = train_ds.class_names
    if actual != expected:
        raise ValueError(
            "Emotion folders must be named exactly: angry, disgust, fear, happy, neutral, sad, surprise. "
            f"Found: {actual}"
        )

    # Keras alphabetical folder order differs from the display order used by app.py.
    # Keep the model's class order alphabetical to match dataset labels.
    return train_ds, test_ds, actual


def build_model():
    model = models.Sequential([
        layers.Input(shape=(48, 48, 1)),
        layers.Rescaling(1.0 / 255.0),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(7, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    train_ds, test_ds, class_names = load_fer2013_from_directory()
    model = build_model()
    model.summary()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True),
    ]
    model.fit(train_ds, validation_data=test_ds, epochs=20, callbacks=callbacks)
    loss, accuracy = model.evaluate(test_ds, verbose=0)
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Saved model to: {MODEL_PATH}")
