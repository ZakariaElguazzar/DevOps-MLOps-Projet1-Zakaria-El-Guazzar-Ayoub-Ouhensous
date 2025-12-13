# prepare_data.py
# =========================================
# Data preparation for CIFAR-100 (30 classes)
# =========================================

import pickle
import numpy as np
import tensorflow as tf
import mlflow
import yaml


from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# --------------------
# CONFIG
# --------------------
# Charger les paramètres
with open("params.yaml", 'r') as f:
    params = yaml.safe_load(f)
    augment = params['prepare']['augment']
IMG_SIZE = (params['prepare']['img_size'] , params['prepare']['img_size'])
BATCH_SIZE = params['prepare']['batch_size']

DATA_PATH = "data/cifar100.npz"
META_PATH = "data/meta"

SELECTED_CLASSES = [
    'bottle','bowl','can','cup','plate',
    'apple','mushroom','orange','pear','sweet_pepper',
    'clock','keyboard','lamp','telephone','television',
    'bed','chair','couch','table','wardrobe',
    'bicycle','bus','motorcycle','pickup_truck','train',
    'lawn_mower','rocket','streetcar','tank','tractor'
]


# --------------------
# LOAD DATA
# --------------------
def load_cifar100():
    data = np.load(DATA_PATH)
    return data['x_train'], data['y_train'].flatten(), \
           data['x_test'], data['y_test'].flatten()


def load_meta():
    with open(META_PATH, 'rb') as f:
        meta = pickle.load(f, encoding='latin1')
    return meta['fine_label_names']


# --------------------
# PREPARE DATA
# --------------------
def prepare_data():
    x_train, y_train, x_test, y_test = load_cifar100()
    class_names = load_meta()

    print(x_train[0])

    selected_indices = [class_names.index(c) for c in SELECTED_CLASSES]

    mask_train = np.isin(y_train, selected_indices)
    mask_test = np.isin(y_test, selected_indices)

    x_train, y_train = x_train[mask_train], y_train[mask_train]
    x_test, y_test = x_test[mask_test], y_test[mask_test]

    # Remap labels
    label_map = {old: i for i, old in enumerate(selected_indices)}
    y_train = np.array([label_map[y] for y in y_train])
    y_test = np.array([label_map[y] for y in y_test])

    # Train / Val split
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.1,
        random_state=42, stratify=y_train
    )

    # Resize
    x_train = tf.image.resize(x_train*255, (128, 128), method='bicubic').numpy().astype(np.uint8)
    x_val = tf.image.resize(x_val*255, (128, 128), method='bicubic').numpy().astype(np.uint8)
    x_test = tf.image.resize(x_test*255, (128, 128), method='bicubic').numpy().astype(np.uint8)

    print("Data prepared:")
    print(x_train[0])

    print(f"Train size: {x_train.shape}")
    print(f"Val size: {x_val.shape}")
    print(f"Test size: {x_test.shape}")
    print(f"Num classes: {len(SELECTED_CLASSES)}")
    print(f"Image size: {IMG_SIZE}")

    return x_train, y_train, x_val, y_val, x_test, y_test



# --------------------
# MAIN
# --------------------
if __name__ == "__main__":

    #mlflow.start_run(run_name="Dataset_Preparation")

    x_train, y_train, x_val, y_val, x_test, y_test = prepare_data()

    #mlflow.log_param("train_size", len(x_train))
    #mlflow.log_param("val_size", len(x_val))
    #mlflow.log_param("test_size", len(x_test))
    #mlflow.log_param("num_classes", len(SELECTED_CLASSES))

    np.savez(
        "data/processed_data.npz",
        x_train=x_train, y_train=y_train,
        x_val=x_val, y_val=y_val,
        x_test=x_test, y_test=y_test
    )

    #mlflow.log_artifact("data/processed_data.npz")

    #mlflow.end_run()
