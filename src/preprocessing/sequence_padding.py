
# ==========================================================
# FILE 8: SEQUENCE PADDING
# Project: Emotion Detection Using NLP and Deep Learning
# Purpose: Make all numerical sequences equal in length
# ==========================================================


# ----------------------------------------------------------
# PART 1: IMPORT LIBRARIES
# ----------------------------------------------------------

import pandas as pd
import yaml
import pickle
import numpy as np

from pathlib import Path
from tensorflow.keras.preprocessing.sequence import pad_sequences


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
# PART 3: LOAD CLEANED DATASETS
# ----------------------------------------------------------

def load_cleaned_datasets(config):

    print("\nLoading cleaned datasets...")

    data_dir = (
        PROJECT_ROOT
        / config["data"]["processed_dir"]
        / "cleaned"
    )

    train_df = pd.read_csv(data_dir / "train_cleaned.csv")
    validation_df = pd.read_csv(data_dir / "validation_cleaned.csv")
    test_df = pd.read_csv(data_dir / "test_cleaned.csv")

    print("Training records:", len(train_df))
    print("Validation records:", len(validation_df))
    print("Testing records:", len(test_df))

    return train_df, validation_df, test_df


# ----------------------------------------------------------
# PART 4: LOAD SAVED TOKENIZER
# ----------------------------------------------------------

def load_tokenizer(config):

    print("\nLoading saved tokenizer...")

    tokenizer_path = (
        PROJECT_ROOT
        / config["artifacts"]["tokenizer_dir"]
        / config["artifacts"]["tokenizer_file"]
    )

    if not tokenizer_path.exists():
        raise FileNotFoundError(
            "Tokenizer file not found. Run tokenizer.py first."
        )

    with open(tokenizer_path, "rb") as file:
        tokenizer = pickle.load(file)

    print("Tokenizer loaded successfully.")

    return tokenizer


# ----------------------------------------------------------
# PART 5: CONVERT TEXT INTO SEQUENCES
# ----------------------------------------------------------

def convert_text_to_sequences(
    train_df,
    validation_df,
    test_df,
    tokenizer,
    config
):

    print("\nConverting text into numerical sequences...")

    text_column = config["data"]["text_column"]

    train_sequences = tokenizer.texts_to_sequences(
        train_df[text_column].astype(str).tolist()
    )

    validation_sequences = tokenizer.texts_to_sequences(
        validation_df[text_column].astype(str).tolist()
    )

    test_sequences = tokenizer.texts_to_sequences(
        test_df[text_column].astype(str).tolist()
    )

    return (
        train_sequences,
        validation_sequences,
        test_sequences
    )


# ----------------------------------------------------------
# PART 6: APPLY PADDING AND TRUNCATION
# ----------------------------------------------------------

def apply_padding(
    train_sequences,
    validation_sequences,
    test_sequences,
    config
):

    print("\nApplying padding and truncation...")

    preprocessing = config["preprocessing"]

    max_length = preprocessing["max_sequence_length"]

    padding_type = preprocessing["padding_type"]

    truncating_type = preprocessing["truncating_type"]

    train_padded = pad_sequences(
        train_sequences,
        maxlen=max_length,
        padding=padding_type,
        truncating=truncating_type
    )

    validation_padded = pad_sequences(
        validation_sequences,
        maxlen=max_length,
        padding=padding_type,
        truncating=truncating_type
    )

    test_padded = pad_sequences(
        test_sequences,
        maxlen=max_length,
        padding=padding_type,
        truncating=truncating_type
    )

    print("Padding completed successfully.")

    return train_padded, validation_padded, test_padded


# ----------------------------------------------------------
# PART 7: EXTRACT LABELS
# ----------------------------------------------------------

def extract_labels(
    train_df,
    validation_df,
    test_df,
    config
):

    label_column = config["data"]["label_column"]

    y_train = train_df[label_column].to_numpy(dtype=np.int64)

    y_validation = validation_df[label_column].to_numpy(dtype=np.int64)

    y_test = test_df[label_column].to_numpy(dtype=np.int64)

    print("\nEmotion labels extracted successfully.")

    return y_train, y_validation, y_test


# ----------------------------------------------------------
# PART 8: SAVE PADDED DATASETS
# ----------------------------------------------------------

def save_padded_datasets(
    train_padded,
    validation_padded,
    test_padded,
    y_train,
    y_validation,
    y_test
):

    print("\nSaving padded datasets...")

    output_dir = PROJECT_ROOT / "data" / "processed" / "padded"

    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "X_train.npy", train_padded)

    np.save(output_dir / "X_validation.npy", validation_padded)

    np.save(output_dir / "X_test.npy", test_padded)

    np.save(output_dir / "y_train.npy", y_train)

    np.save(output_dir / "y_validation.npy", y_validation)

    np.save(output_dir / "y_test.npy", y_test)

    print("All padded datasets saved successfully.")

    print("Output directory:", output_dir)


# ----------------------------------------------------------
# PART 9: DISPLAY SAMPLE OUTPUT
# ----------------------------------------------------------

def display_samples(train_sequences, train_padded):

    print("\nPadding example:")

    print("\nOriginal sequence:")

    print(train_sequences[0])

    print("\nPadded sequence:")

    print(train_padded[0])

    print("\nPadded dataset shape:")

    print(train_padded.shape)


# ----------------------------------------------------------
# PART 10: MAIN FUNCTION
# ----------------------------------------------------------

def main():

    print("=" * 55)

    print("EMOTION DETECTION - SEQUENCE PADDING")

    print("=" * 55)

    config = load_config()

    train_df, validation_df, test_df = load_cleaned_datasets(config)

    tokenizer = load_tokenizer(config)

    (
        train_sequences,
        validation_sequences,
        test_sequences
    ) = convert_text_to_sequences(
        train_df,
        validation_df,
        test_df,
        tokenizer,
        config
    )

    (
        train_padded,
        validation_padded,
        test_padded
    ) = apply_padding(
        train_sequences,
        validation_sequences,
        test_sequences,
        config
    )

    (
        y_train,
        y_validation,
        y_test
    ) = extract_labels(
        train_df,
        validation_df,
        test_df,
        config
    )

    display_samples(
        train_sequences,
        train_padded
    )

    save_padded_datasets(
        train_padded,
        validation_padded,
        test_padded,
        y_train,
        y_validation,
        y_test
    )

    print("\nDataset shapes:")

    print("X_train:", train_padded.shape)
    print("X_validation:", validation_padded.shape)
    print("X_test:", test_padded.shape)

    print("\nLabel shapes:")

    print("y_train:", y_train.shape)
    print("y_validation:", y_validation.shape)
    print("y_test:", y_test.shape)

    print("\n" + "=" * 55)

    print("SEQUENCE PADDING COMPLETED SUCCESSFULLY")

    print("=" * 55)


# ----------------------------------------------------------
# PART 11: EXECUTE PROGRAM
# ----------------------------------------------------------

if __name__ == "__main__":

    main()
