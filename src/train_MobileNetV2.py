# train_MobileNetV2.py
# =========================================
# MobileNetV2 Training (Phase 1 + FT)
# =========================================

import mlflow
import mlflow.keras
import mlflow.sklearn
import numpy as np
import tensorflow as tf
import yaml
import json

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

mlflow.keras.autolog()      # Track automatique Keras
mlflow.sklearn.autolog()    # Track automatique sklearn


# --------------------
# CONFIG
# --------------------
with open("params.yaml", 'r') as f:
    params = yaml.safe_load(f)
    NUM_CLASSES = 30
    INPUT_SHAPE = (128, 128, 3)
    BATCH_SIZE = params['train']['batch_size']
    EPOCHS_PHASE1 = params['train']['epochs']
    EPOCHS_PHASE2 = params['train']['epochs']
    DATA_PATH = "data/processed_data.npz"
    MODEL_DIR = "models/"
    METRICS_DIR = "metrics/"


print(BATCH_SIZE, INPUT_SHAPE, EPOCHS_PHASE1, EPOCHS_PHASE2)




# --------------------
# LOAD DATA
# --------------------
data = np.load(DATA_PATH)

x_train, y_train = data["x_train"], data["y_train"]
x_val, y_val = data["x_val"], data["y_val"]

train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2]
    )

val_test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

train_gen = train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)
val_gen = val_test_datagen.flow(x_val, y_val, batch_size=BATCH_SIZE, shuffle=False)


# --------------------
# MODEL
# --------------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=INPUT_SHAPE
)
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation="relu", kernel_regularizer=l2(1e-4))(x)
x = Dropout(0.5)(x)
x = Dense(256, activation="relu", kernel_regularizer=l2(1e-4))(x)
x = Dropout(0.3)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(base_model.input, outputs)

model.compile(
    optimizer=Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# --------------------
# CALLBACKS
# --------------------
callbacks_bf_ft = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(patience=2, factor=0.2),
    ModelCheckpoint(
        MODEL_DIR + "MobileNetV2_bf_ft.keras",
        save_best_only=False,
        save_freq="epoch"
    )
]
callbacks_ft = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(patience=2, factor=0.2),
    ModelCheckpoint(
        MODEL_DIR + "MobileNetV2_ft.keras",
        save_best_only=False,
        save_freq="epoch"
    )
]


# --------------------
# PHASE 1 – Avant Fine-Tuning
# --------------------
with mlflow.start_run(run_name="MobileNetV2_Phase1"):

    # Log params
    mlflow.log_params({
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS_PHASE1,
        "learning_rate": 1e-4,
        "phase": "before_finetune"
    })

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_PHASE1,
        callbacks=callbacks_bf_ft
    )

    # Log metrics train + val
    metrics_bf_ft = {
        "train_loss": float(history.history['loss'][-1]),
        "train_accuracy": float(history.history['accuracy'][-1]),
        "val_loss": float(history.history['val_loss'][-1]),
        "val_accuracy": float(history.history['val_accuracy'][-1])
    }
    mlflow.log_metrics(metrics_bf_ft)

    # Sauvegarde modèle
    model.save(MODEL_DIR + "MobileNetV2_bf_ft.keras")
    mlflow.keras.log_model(model, "MobileNetV2_bf_ft")

    # Sauvegarde JSON
    with open(METRICS_DIR + "train_MobileNetV2_bf_ft_metrics.json", "w") as f:
        json.dump(metrics_bf_ft, f)


# --------------------
# PHASE 2 – Fine-Tuning
# --------------------
with mlflow.start_run(run_name="MobileNetV2_FineTuning"):

    # Reload model
    model = load_model(MODEL_DIR + "MobileNetV2_bf_ft.keras")

    # Définir les couches à fine-tuner
    fine_tune_at = len(model.layers) - 90
    for layer in model.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in model.layers[fine_tune_at:]:
        layer.trainable = True

    model.compile(
        optimizer=Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    mlflow.log_params({
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS_PHASE2,
        "learning_rate": 1e-4,
        "phase": "finetune"
    })

    history_ft = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_PHASE2,
        callbacks=callbacks_ft
    )

    # Log metrics train + val
    metrics_ft = {
        "train_loss": float(history_ft.history['loss'][-1]),
        "train_accuracy": float(history_ft.history['accuracy'][-1]),
        "val_loss": float(history_ft.history['val_loss'][-1]),
        "val_accuracy": float(history_ft.history['val_accuracy'][-1])
    }
    mlflow.log_metrics(metrics_ft)

    # Sauvegarde modèle
    model.save(MODEL_DIR + "MobileNetV2_ft.keras")
    mlflow.keras.log_model(model, "MobileNetV2_ft")

    # Sauvegarde JSON
    with open(METRICS_DIR + "train_MobileNetV2_ft_metrics.json", "w") as f:
        json.dump(metrics_ft, f)
