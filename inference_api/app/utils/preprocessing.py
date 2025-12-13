# app/utils/preprocessing.py

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

IMG_SIZE = (128, 128, 3)

def preprocess_cnn(image: Image.Image):
    image = image.resize(IMG_SIZE)
    x = np.array(image).astype("float32") / 255.0
    return np.expand_dims(x, axis=0)

def preprocess_mobilenet(image: Image.Image):
    image = image.resize(IMG_SIZE)
    x = np.array(image).astype("float32")
    x = mobilenet_preprocess(x)
    return np.expand_dims(x, axis=0)

