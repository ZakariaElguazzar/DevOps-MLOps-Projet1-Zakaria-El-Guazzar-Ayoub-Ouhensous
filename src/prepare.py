# src/prepare.py (squelette)
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse
import yaml
import shutil

def main(params_path="params.yaml"):
    with open(params_path) as f:
        params = yaml.safe_load(f)
    img_size = params['prepare']['img_size']
    augment = params['prepare']['augment']

    # Exemple rapide : utiliser cifar10 via tf.keras.datasets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # convert to folders or save as tfrecords etc. For simplicity, create npz
    os.makedirs("data", exist_ok=True)
    import numpy as np
    np.savez("data/cifar10.npz", x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    print("Saved data/cifar10.npz")

if __name__ == "__main__":
    main()

