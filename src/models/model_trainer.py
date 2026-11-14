
import json
import yaml
import numpy as np

from pathlib import Path
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ==========================================
# PART 1: LOAD CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

CONFIG_PATH = BASE_DIR / "config" / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

print("Configuration loaded successfully.")


# ==========================================
# PART 2: LOAD TRAINING DATA
# ==========================================

PADDED_DIR = BASE_DIR / "data" / "processed" / "padded"

X_train = np.load(PADDED_DIR / "X_train.npy")
X_validation = np.load(PADDED_DIR / "X_validation.npy")

y_train = np.load(PADDED_DIR / "y_train.npy")
y_validation = np.load(PADDED_DIR / "y_validation.npy")

print("\nTraining data loaded successfully.")

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("X_validation shape:", X_validation.shape)
print("y_validation shape:", y_validation.shape)


# ==========================================
# PART 3: VALIDATE DATA
# ==========================================

if X_train.ndim != 2 or X_validation.ndim != 2:
    raise ValueError("Input data must be two-dimensional.")

if len(X_train) != len(y_train):
    raise ValueError("Training input and labels have different lengths.")

if len(X_validation) != len(y_validation):
    raise ValueError("Validation input and labels have different lengths.")

if not np.issubdtype(y_train.dtype, np.integer):
    raise ValueError("Training labels must contain integer class IDs.")

if not np.issubdtype(y_validation.dtype, np.integer):
    raise ValueError("Validation labels must contain integer class IDs.")

num_classes = config["labels"]["num_classes"]

if np.any(y_train < 0) or np.any(y_train >= num_classes):
    raise ValueError("Invalid training label values.")

if np.any(y_validation < 0) or np.any(y_validation >= num_classes):
    raise ValueError("Invalid validation label values.")

if X_train.shape[1] != config["preprocessing"]["max_sequence_length"]:
    raise ValueError("Training sequence length does not match config.")

if X_validation.shape[1] != config["preprocessing"]["max_sequence_length"]:
    raise ValueError("Validation sequence length does not match config.")

print("Dataset validation completed successfully.")


# ==========================================
# PART 4: LOAD MODEL ARCHITECTURE
# ==========================================

MODEL_DIR = BASE_DIR / "artifacts" / "models"

ARCHITECTURE_PATH = MODEL_DIR / "model_architecture.json"

if not ARCHITECTURE_PATH.exists():
    raise FileNotFoundError(
        "Model architecture not found. Execute File 12 first."
    )

with open(ARCHITECTURE_PATH, "r", encoding="utf-8") as file:
    model_json = file.read()

model = model_from_json(model_json)

print("\nLSTM model architecture loaded successfully.")


# ==========================================
# PART 5: COMPILE MODEL
# ==========================================

from tensorflow.keras.optimizers import Adam

learning_rate = config["training"]["learning_rate"]

model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss=config["model"]["loss"],
    metrics=["accuracy"]
)

print("Model compiled successfully.")


# ==========================================
# PART 6: CREATE TRAINING CALLBACKS
# ==========================================

training_config = config["training"]
early_stopping_config = training_config["early_stopping"]

early_stopping = EarlyStopping(
    monitor=early_stopping_config["monitor"],
    patience=early_stopping_config["patience"],
    restore_best_weights=early_stopping_config["restore_best_weights"],
    verbose=1
)

MODEL_PATH = MODEL_DIR / config["artifacts"]["model_file"]

checkpoint = ModelCheckpoint(
    filepath=str(MODEL_PATH),
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

print("Training callbacks created successfully.")


# ==========================================
# PART 7: TRAIN MODEL
# ==========================================

epochs = training_config["epochs"]
batch_size = training_config["batch_size"]

print("\nStarting LSTM model training...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_validation, y_validation),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stopping, checkpoint],
    verbose=1
)


# ==========================================
# PART 8: SAVE TRAINING HISTORY
# ==========================================

history_path = MODEL_DIR / "training_history.json"

history_data = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}

with open(history_path, "w", encoding="utf-8") as file:
    json.dump(history_data, file, indent=4)

print("\nTraining history saved successfully.")


# ==========================================
# PART 9: DISPLAY TRAINING RESULTS
# ==========================================

print("\nTraining Results:")

for epoch in range(len(history.history["loss"])):
    print(f"\nEpoch {epoch + 1}")
    print("Training Loss:", round(history.history["loss"][epoch], 4))
    print("Training Accuracy:", round(history.history["accuracy"][epoch], 4))
    print("Validation Loss:", round(history.history["val_loss"][epoch], 4))
    print("Validation Accuracy:", round(history.history["val_accuracy"][epoch], 4))


print("\nModel training completed successfully!")
print("Best model saved at:", MODEL_PATH)
print("Training history saved at:", history_path)
