# src/train.py (modifié)
import tensorflow as tf
import json, os, yaml
import numpy as np

tf.config.set_visible_devices([], 'GPU')
print("GPU disabled → TensorFlow will run on CPU.")

def build_simple_cnn(input_shape, num_classes):
    from tensorflow.keras import layers, models
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32,3,activation='relu'), layers.MaxPool2D(),
        layers.Conv2D(64,3,activation='relu'), layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main(params_path="params.yaml"):
    with open(params_path) as f:
        params = yaml.safe_load(f)

    epochs = params['train']['epochs']
    batch_size = params['train']['batch_size']

    # load data
    data = np.load("data/cifar100.npz")
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    model = build_simple_cnn(x_train.shape[1:], 100)

    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size
    )

    # Enregistrer le modèle
    os.makedirs("models", exist_ok=True)
    model.save("models/model.h5")

    # === Nouveau dossier metrics ===
    os.makedirs("metrics", exist_ok=True)

    train_metrics = {
        "val_loss": float(history.history['val_loss'][-1]),
        "val_accuracy": float(history.history['val_accuracy'][-1])
    }

    # Enregistrement dans metrics/train_metrics.json
    with open("metrics/train_metrics.json", "w") as f:
        json.dump(train_metrics, f)

    print("Saved model and metrics/train_metrics.json")

if __name__ == "__main__":
    main()

