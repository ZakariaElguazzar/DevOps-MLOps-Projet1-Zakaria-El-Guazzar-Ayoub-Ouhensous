# src/evaluate_MobileNetV2.py
# =========================================
# Evaluation script for MobileNetV2
# =========================================

import os
import json
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.keras
import mlflow.sklearn
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Fix for environments without GUI (server / CI)
plt.switch_backend("Agg")

mlflow.keras.autolog()      # Track automatique Keras
mlflow.sklearn.autolog()    # Track automatique sklearn


# --------------------
# CONFIG
# --------------------

DATA_PATH = "data/processed_data.npz"
MODEL_PATH = "models/CNN_simple.keras"

PLOTS_DIR = "plots/"
METRICS_DIR = "metrics/"
# --------------------
# MAIN
# --------------------
def main():

    mlflow.start_run(run_name="Evaluate_CNN_simple")

    # ---- Load data ----
    data = np.load(DATA_PATH)
    x_test = data["x_test"]
    y_test = data["y_test"]

    # MobileNet preprocessing
    x_test = preprocess_input(x_test.astype("float32"))

    # ---- Load model ----
    model = tf.keras.models.load_model(MODEL_PATH)

    test_loss,test_acc =model.evaluate(x_test, y_test, batch_size=32, verbose=2)

    mlflow.log_metric("test_acc", test_acc)
    mlflow.log_metric("test_loss", test_loss)

    # ---- Prediction ----
    y_pred_probs = model.predict(x_test, batch_size=32)
    y_pred = y_pred_probs.argmax(axis=1)

    # ---- Metrics ----
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # ---- Save metrics ----
    metrics = {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),

    }

    with open(METRICS_DIR + "evaluate_CNN_simple_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    with open(METRICS_DIR + "classification_report_CNN_simple.json", "w") as f:
        json.dump(report, f, indent=4)

    mlflow.log_artifact(METRICS_DIR + "classification_report_CNN_simple.json", "classification_report")

    # ---- Confusion Matrix Plot ----
    plt.figure(figsize=(10, 8))
    plt.imshow(cm)
    plt.title("Confusion Matrix - MobileNetV2")
    plt.colorbar()
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR + "confusion_CNN_simple.png", dpi=300)
    plt.close()

    print("✅ Evaluation completed")
    print(f"📊 Test Accuracy: {acc:.4f}")
    print(f"📊 Test Loss: {test_loss:.4f}")

    print("📁 Saved:")
    print(" - plots/confusion_CNN_simple.png")
    print(" - metrics/evaluate_CNN_simple_metrics.json")
    print(" - metrics/classification_report_CNN_simple.json")

    mlflow.log_artifact(PLOTS_DIR + "confusion_CNN_simple.png", "confusion_matrix")
    mlflow.end_run()


# --------------------
# ENTRY POINT
# --------------------
if __name__ == "__main__":
    main()
