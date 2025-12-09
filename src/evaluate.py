# src/evaluate.py (corrigé)
import tensorflow as tf, numpy as np, json, os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

plt.switch_backend("Agg")   # <-- AJOUTER CETTE LIGNE (fix plots without GUI)

def main():
    data = np.load("data/cifar100.npz")
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
    evaluate_metrics = {"test_accuracy": float(accuracy_score(y_test.flatten(), preds))}
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/evaluate_metrics.json", "w") as f:
        json.dump(evaluate_metrics, f)
    print("Saved plots/confusion.png and evaluate_metrics.json")

if __name__ == "__main__":
    main()

