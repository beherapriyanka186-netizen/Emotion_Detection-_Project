
# ==========================================================
# FILE 11: EMBEDDING PROCESSOR
# Project: Emotion Detection Using NLP and Deep Learning
# Purpose: Prepare numerical sequences for the embedding layer
# ==========================================================


# ----------------------------------------------------------
# PART 1: IMPORT LIBRARIES
# ----------------------------------------------------------

import numpy as np
import pickle
import json
import yaml

from pathlib import Path


# ----------------------------------------------------------
# PART 2: PROJECT PATH AND CONFIGURATION
# ----------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config():

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:

        config = yaml.safe_load(file)

    return config


# ----------------------------------------------------------
# PART 3: LOAD PADDED DATASETS
# ----------------------------------------------------------

def load_padded_datasets():

    print("\nLoading padded datasets...")

    data_dir = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "padded"
    )

    required_files = [
        "X_train.npy",
        "X_validation.npy",
        "X_test.npy",
        "y_train.npy",
        "y_validation.npy",
        "y_test.npy"
    ]

    for filename in required_files:

        file_path = data_dir / filename

        if not file_path.exists():

            raise FileNotFoundError(
                f"{filename} not found. "
                "Run sequence_padding.py first."
            )

    X_train = np.load(data_dir / "X_train.npy")
    X_validation = np.load(data_dir / "X_validation.npy")
    X_test = np.load(data_dir / "X_test.npy")

    y_train = np.load(data_dir / "y_train.npy")
    y_validation = np.load(data_dir / "y_validation.npy")
    y_test = np.load(data_dir / "y_test.npy")

    print("Padded datasets loaded successfully.")

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test
    )


# ----------------------------------------------------------
# PART 4: LOAD TOKENIZER
# ----------------------------------------------------------

def load_tokenizer(config):

    print("\nLoading tokenizer...")

    tokenizer_path = (
        PROJECT_ROOT
        / config["artifacts"]["tokenizer_dir"]
        / config["artifacts"]["tokenizer_file"]
    )

    if not tokenizer_path.exists():

        raise FileNotFoundError(
            "Tokenizer not found. Run tokenizer.py first."
        )

    with open(tokenizer_path, "rb") as file:

        tokenizer = pickle.load(file)

    print("Tokenizer loaded successfully.")

    return tokenizer


# ----------------------------------------------------------
# PART 5: LOAD VOCABULARY
# ----------------------------------------------------------

def load_vocabulary(config):

    print("\nLoading vocabulary...")

    vocabulary_path = (
        PROJECT_ROOT
        / "artifacts"
        / "vocabulary"
        / "vocabulary.json"
    )

    if not vocabulary_path.exists():

        raise FileNotFoundError(
            "Vocabulary not found. Run vocabulary_builder.py first."
        )

    with open(
        vocabulary_path,
        "r",
        encoding="utf-8"
    ) as file:

        vocabulary = json.load(file)

    print("Vocabulary loaded successfully.")

    return vocabulary


# ----------------------------------------------------------
# PART 6: VERIFY INPUT SEQUENCES
# ----------------------------------------------------------

def verify_sequences(
    X_train,
    X_validation,
    X_test,
    config
):

    print("\nVerifying numerical sequences...")

    max_length = config["preprocessing"]["max_sequence_length"]

    datasets = {
        "Training": X_train,
        "Validation": X_validation,
        "Testing": X_test
    }

    for name, data in datasets.items():

        if data.ndim != 2:

            raise ValueError(
                f"{name} input must be a 2D array."
            )

        if data.shape[1] != max_length:

            raise ValueError(
                f"{name} sequence length is incorrect."
            )

        if not np.issubdtype(data.dtype, np.integer):

            raise ValueError(
                f"{name} sequences must contain integer IDs."
            )

        if np.any(data < 0):

            raise ValueError(
                f"Negative token IDs found in {name} dataset."
            )

        print(
            f"{name} sequences verified: {data.shape}"
        )

    print("All numerical sequences verified successfully.")


# ----------------------------------------------------------
# PART 7: VERIFY VOCABULARY SIZE
# ----------------------------------------------------------

def verify_vocabulary(tokenizer, vocabulary, config):

    print("\nVerifying vocabulary information...")

    max_words = config["preprocessing"]["max_words"]

    word_index = tokenizer.word_index

    oov_token = config["preprocessing"]["oov_token"]

    oov_index = word_index.get(oov_token)

    if oov_index is None:

        raise ValueError("OOV token is missing.")

    if len(vocabulary) == 0:

        raise ValueError("Vocabulary is empty.")

    if len(vocabulary) > max_words - 1:

        raise ValueError("Vocabulary exceeds configured limit.")

    print("Vocabulary size:", len(vocabulary))

    print("Maximum words:", max_words)

    print("OOV token:", oov_token)

    print("OOV index:", oov_index)

    print("Vocabulary verification successful.")


# ----------------------------------------------------------
# PART 8: CHECK LABEL DATASETS
# ----------------------------------------------------------

def verify_labels(
    y_train,
    y_validation,
    y_test,
    config
):

    print("\nVerifying emotion labels...")

    num_classes = config["labels"]["num_classes"]

    datasets = {
        "Training": y_train,
        "Validation": y_validation,
        "Testing": y_test
    }

    for name, labels in datasets.items():

        if labels.ndim != 1:

            raise ValueError(
                f"{name} labels must be one-dimensional."
            )

        if len(labels) == 0:

            raise ValueError(
                f"{name} labels are empty."
            )

        if np.any(labels < 0) or np.any(labels >= num_classes):

            raise ValueError(
                f"Invalid emotion labels in {name} dataset."
            )

        print(
            f"{name} labels verified: {labels.shape}"
        )


# ----------------------------------------------------------
# PART 9: PREPARE EMBEDDING INPUT INFORMATION
# ----------------------------------------------------------

def prepare_embedding_information(config, tokenizer):

    print("\nPreparing embedding information...")

    max_words = config["preprocessing"]["max_words"]

    embedding_dim = config["model"]["embedding_dim"]

    # Token IDs range from 0 up to max_words - 1.
    input_dim = min(
        len(tokenizer.word_index) + 1,
        max_words
    )

    embedding_information = {
        "input_dim": input_dim,
        "output_dim": embedding_dim,
        "input_length": config["preprocessing"]["max_sequence_length"],
        "mask_zero": True
    }

    print("Embedding input dimension:", input_dim)

    print("Embedding output dimension:", embedding_dim)

    print(
        "Maximum sequence length:",
        embedding_information["input_length"]
    )

    return embedding_information


# ----------------------------------------------------------
# PART 10: SAVE EMBEDDING INFORMATION
# ----------------------------------------------------------

def save_embedding_information(embedding_information):

    print("\nSaving embedding information...")

    output_dir = PROJECT_ROOT / "artifacts" / "embeddings"

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "embedding_config.json"

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            embedding_information,
            file,
            indent=4
        )

    print("Embedding information saved successfully.")

    print("Saved location:", output_path)


# ----------------------------------------------------------
# PART 11: MAIN FUNCTION
# ----------------------------------------------------------

def main():

    print("=" * 55)

    print("EMOTION DETECTION - EMBEDDING PROCESSOR")

    print("=" * 55)

    config = load_config()

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test
    ) = load_padded_datasets()

    tokenizer = load_tokenizer(config)

    vocabulary = load_vocabulary(config)

    verify_sequences(
        X_train,
        X_validation,
        X_test,
        config
    )

    verify_vocabulary(
        tokenizer,
        vocabulary,
        config
    )

    verify_labels(
        y_train,
        y_validation,
        y_test,
        config
    )

    embedding_information = prepare_embedding_information(
        config,
        tokenizer
    )

    save_embedding_information(
        embedding_information
    )

    print("\nDataset information:")

    print("X_train:", X_train.shape)
    print("X_validation:", X_validation.shape)
    print("X_test:", X_test.shape)

    print("\n" + "=" * 55)

    print("EMBEDDING PROCESSING COMPLETED SUCCESSFULLY")

    print("=" * 55)


# ----------------------------------------------------------
# PART 12: EXECUTE PROGRAM
# ----------------------------------------------------------

if __name__ == "__main__":

    main()
