
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from tensorflow.keras.models import load_model

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# PART 1: LOAD CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

CONFIG_PATH = BASE_DIR / "config" / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

print("Configuration loaded successfully.")


# ==========================================
# PART 2: LOAD TEST DATA
# ==========================================

PADDED_DIR = BASE_DIR / "data" / "processed" / "padded"

X_test = np.load(PADDED_DIR / "X_test.npy")
y_test = np.load(PADDED_DIR / "y_test.npy")

print("\nTest dataset loaded successfully.")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ==========================================
# PART 3: VALIDATE TEST DATA
# ==========================================

if X_test.ndim != 2:
    raise ValueError("Test input must be a 2D array.")

if len(X_test) != len(y_test):
    raise ValueError("Test inputs and labels have different lengths.")

num_classes = config["labels"]["num_classes"]

if np.any(y_test < 0) or np.any(y_test >= num_classes):
    raise ValueError("Invalid test label values.")

if X_test.shape[1] != config["preprocessing"]["max_sequence_length"]:
    raise ValueError("Test sequence length does not match config.")

print("Test data validation completed successfully.")


# ==========================================
# PART 4: LOAD TRAINED MODEL
# ==========================================

MODEL_DIR = BASE_DIR / "artifacts" / "models"

MODEL_PATH = MODEL_DIR / config["artifacts"]["model_file"]

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Trained model not found. Execute File 13 first."
    )

model = load_model(MODEL_PATH)

print("\nTrained LSTM model loaded successfully.")


# ==========================================
# PART 5: PREDICT TEST DATA
# ==========================================

print("\nPredicting emotions for test data...")

y_probabilities = model.predict(
    X_test,
    batch_size=config["training"]["batch_size"],
    verbose=1
)

y_pred = np.argmax(y_probabilities, axis=1)

print("Test predictions completed successfully.")


# ==========================================
# PART 6: CALCULATE EVALUATION METRICS
# ==========================================

class_names = config["labels"]["classes"]

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\n========== MODEL EVALUATION RESULTS ==========")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))


# ==========================================
# PART 7: GENERATE CLASSIFICATION REPORT
# ==========================================

report = classification_report(
    y_test,
    y_pred,
    labels=list(range(num_classes)),
    target_names=class_names,
    zero_division=0
)

print("\nClassification Report:")
print(report)


# ==========================================
# PART 8: CREATE OUTPUT DIRECTORIES
# ==========================================

METRICS_DIR = BASE_DIR / "artifacts" / "metrics"

FIGURES_DIR = BASE_DIR / "reports" / "figures"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# PART 9: SAVE EVALUATION METRICS
# ==========================================

evaluation_results = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "total_test_samples": int(len(y_test)),
    "emotion_classes": class_names
}

METRICS_PATH = METRICS_DIR / "model_evaluation.json"

with open(METRICS_PATH, "w", encoding="utf-8") as file:
    json.dump(evaluation_results, file, indent=4)


# Save classification report

REPORT_PATH = METRICS_DIR / "classification_report.txt"

with open(REPORT_PATH, "w", encoding="utf-8") as file:
    file.write(report)

print("\nEvaluation metrics saved successfully.")


# ==========================================
# PART 10: GENERATE CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=list(range(num_classes))
)

plt.figure(figsize=(9, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("LSTM Emotion Detection - Confusion Matrix")
plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")

plt.tight_layout()

CONFUSION_PATH = FIGURES_DIR / "confusion_matrix.png"

plt.savefig(CONFUSION_PATH, dpi=300)
plt.show()

plt.close()

print("Confusion matrix saved successfully.")


# ==========================================
# PART 11: GENERATE ACCURACY GRAPH
# ==========================================

plt.figure(figsize=(7, 5))

plt.bar(
    ["Accuracy", "Precision", "Recall", "F1 Score"],
    [accuracy, precision, recall, f1]
)

plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("LSTM Model Performance")

plt.tight_layout()

PERFORMANCE_PATH = FIGURES_DIR / "model_performance.png"

plt.savefig(PERFORMANCE_PATH, dpi=300)
plt.show()

plt.close()

print("Model performance graph saved successfully.")


# ==========================================
# PART 12: FINAL OUTPUT
# ==========================================

print("\n========== EVALUATION COMPLETED ==========")

print("Metrics:", METRICS_PATH)
print("Classification report:", REPORT_PATH)
print("Confusion matrix:", CONFUSION_PATH)
print("Performance graph:", PERFORMANCE_PATH)

print("\nFile 14 executed successfully!")
