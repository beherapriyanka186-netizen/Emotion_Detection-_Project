
import json
import yaml
from pathlib import Path

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


# ==========================================
# PART 1: LOAD CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

CONFIG_PATH = BASE_DIR / "config" / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

print("Configuration loaded successfully.")


# ==========================================
# PART 2: LOAD EMBEDDING CONFIGURATION
# ==========================================

EMBEDDING_CONFIG_PATH = (
    BASE_DIR / "artifacts" / "embeddings" / "embedding_config.json"
)

if not EMBEDDING_CONFIG_PATH.exists():
    raise FileNotFoundError(
        "Embedding configuration not found. "
        "Please execute File 11 first."
    )

with open(EMBEDDING_CONFIG_PATH, "r", encoding="utf-8") as file:
    embedding_config = json.load(file)

input_dim = embedding_config["input_dim"]
embedding_dim = embedding_config["output_dim"]
max_sequence_length = embedding_config["input_length"]

print("Embedding configuration loaded successfully.")


# ==========================================
# PART 3: READ MODEL PARAMETERS
# ==========================================

lstm_units = config["model"]["lstm_units"]
dense_units = config["model"]["dense_units"]
dropout_rate = config["model"]["dropout_rate"]

num_classes = config["labels"]["num_classes"]

learning_rate = config["training"]["learning_rate"]

print("\nModel Parameters:")
print("Vocabulary size:", input_dim)
print("Embedding dimension:", embedding_dim)
print("Maximum sequence length:", max_sequence_length)
print("LSTM units:", lstm_units)
print("Dense units:", dense_units)
print("Number of classes:", num_classes)


# ==========================================
# PART 4: BUILD LSTM MODEL
# ==========================================

model = Sequential([
    Input(shape=(max_sequence_length,)),

    Embedding(
        input_dim=input_dim,
        output_dim=embedding_dim,
        mask_zero=True
    ),

    LSTM(lstm_units),

    Dropout(dropout_rate),

    Dense(
        dense_units,
        activation="relu"
    ),

    Dense(
        num_classes,
        activation=config["model"]["activation"]
    )
])


# ==========================================
# PART 5: COMPILE MODEL
# ==========================================

optimizer = Adam(learning_rate=learning_rate)

model.compile(
    optimizer=optimizer,
    loss=config["model"]["loss"],
    metrics=["accuracy"]
)


# ==========================================
# PART 6: DISPLAY MODEL SUMMARY
# ==========================================

print("\nLSTM Model Summary:")

model.summary()


# ==========================================
# PART 7: SAVE MODEL ARCHITECTURE
# ==========================================

MODEL_DIR = BASE_DIR / "artifacts" / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

ARCHITECTURE_PATH = MODEL_DIR / "model_architecture.json"

with open(ARCHITECTURE_PATH, "w", encoding="utf-8") as file:
    file.write(model.to_json())

print("\nModel architecture saved successfully.")
print("Saved location:", ARCHITECTURE_PATH)

print("\nFile 12 executed successfully!")
