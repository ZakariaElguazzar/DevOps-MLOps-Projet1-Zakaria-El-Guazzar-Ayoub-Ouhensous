# src/prepare.py
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse
import yaml
import numpy as np

tf.config.set_visible_devices([], 'GPU')
print("GPU disabled → TensorFlow will run on CPU.")

def main(params_path="params.yaml"):
    # Charger les paramètres
    with open(params_path) as f:
        params = yaml.safe_load(f)

    img_size = params['prepare']['img_size']      # ex: 32
    augment = params['prepare']['augment']        # True / False

    # Charger CIFAR-100
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')

    # Normalisation entre 0 et 1
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Resize si nécessaire
    if img_size != 32:
        x_train = tf.image.resize(x_train, (img_size, img_size)).numpy()
        x_test = tf.image.resize(x_test, (img_size, img_size)).numpy()

    # Augmentation des données (optionnel)
    if augment:
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True
        )
        datagen.fit(x_train)

    # Créer dossier data
    os.makedirs("data", exist_ok=True)

    # Sauvegarde
    np.savez(
        "data/cifar100.npz",
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test
    )

    print("Saved data/cifar100.npz (CIFAR-100 ready)")

if __name__ == "__main__":
    main()

