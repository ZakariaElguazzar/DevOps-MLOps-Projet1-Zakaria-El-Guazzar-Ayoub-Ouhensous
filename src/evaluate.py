# src/evaluate.py (squelette)
import tensorflow as tf, numpy as np, json, os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

def main():
    data = np.load("data/cifar10.npz")
    x_test, y_test = data['x_test'], data['y_test']
    x_test = x_test / 255.0
    model = tf.keras.models.load_model("models/model.h5")
    preds = model.predict(x_test).argmax(axis=1)
    cm = confusion_matrix(y_test.flatten(), preds)
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(8,8))
    plt.imshow(cm)
    plt.title("Confusion matrix")
    plt.colorbar()
    plt.savefig("plots/confusion.png")
    from sklearn.metrics import accuracy_score
    metrics = {"test_accuracy": float(accuracy_score(y_test.flatten(), preds))}
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)
    print("Saved plots/confusion.png and metrics.json")

if __name__ == "__main__":
    main()
