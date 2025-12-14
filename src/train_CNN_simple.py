# train_CNN_simple.py
# =========================================
# Simple CNN Training
# =========================================

# ========================
# Standard library
# ========================
import json
import yaml

# ========================
# Third-party libraries
# ========================
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.keras
import mlflow.sklearn

# ========================
# Keras / TensorFlow
# ========================
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
    DATA_PATH = "data/processed_data.npz"
    MODEL_DIR = "models/"
    METRICS_DIR = "metrics/"

# --------------------
# DATA GENERATORS
# --------------------
# LOAD DATA
# --------------------
data = np.load(DATA_PATH)

x_train, y_train = data["x_train"], data["y_train"]
x_val, y_val = data["x_val"], data["y_val"]

def preprocess_input(x):
    return x / 255.0



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
inputs = Input(shape=INPUT_SHAPE)

# ---- Block 1 ----
x = Conv2D(32, (3,3), padding='same', activation='relu',
           kernel_regularizer=l2(1e-4))(inputs)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)

# ---- Block 2 ----
x = Conv2D(64, (3,3), padding='same', activation='relu',
           kernel_regularizer=l2(1e-4))(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)

# ---- Block 3 ----
x = Conv2D(128, (3,3), padding='same', activation='relu',
           kernel_regularizer=l2(1e-4))(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)

# ---- Block 4 ----
x = Conv2D(256, (3,3), padding='same', activation='relu',
           kernel_regularizer=l2(1e-4))(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)

# ---- Classification Head ----
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
x = Dropout(0.5)(x)

outputs = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# --------------------
# CALLBACKS
# --------------------
callbacks= [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(patience=2, factor=0.2),
    ModelCheckpoint(
        MODEL_DIR + "CNN_simple.keras",
        save_best_only=False,
        save_freq="epoch"
    )
]


# --------------------
# PHASE 1
# --------------------
with mlflow.start_run(run_name="CNN_simple_Training"):

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_PHASE1,
        callbacks=callbacks
    )

    model.save(MODEL_DIR + "CNN_simple.keras")

    train_metrics = {
        "val_loss": float(history.history['val_loss'][-1]),
        "val_accuracy": float(history.history['val_accuracy'][-1]),
        "train_loss": float(history.history['loss'][-1]),
        "train_accuracy": float(history.history['accuracy'][-1])
    }

    mlflow.log_metrics(train_metrics)

    # Enregistrement dans metrics/train_CNN_simple_metrics.json
    with open(METRICS_DIR + "train_CNN_simple_metrics.json", "w") as f:
        json.dump(train_metrics, f)

    mlflow.keras.log_model(model, "CNN_simple")