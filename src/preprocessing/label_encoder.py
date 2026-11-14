
# ==========================================================
# FILE 9: LABEL ENCODER
# Project: Emotion Detection Using NLP and Deep Learning
# Purpose: Preserve and save original emotion label mapping
# ==========================================================

# PART 1: IMPORT LIBRARIES

import numpy as np
import pandas as pd
import yaml
import pickle

from pathlib import Path


# PART 2: PROJECT PATH AND CONFIGURATION

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config():

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


# PART 3: LOAD CLEANED DATASETS

def load_datasets(config):

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


# PART 4: CREATE ORIGINAL LABEL MAPPING

def create_label_mapping(config):

    print("\nCreating original emotion mapping...")

    emotion_classes = config["labels"]["classes"]

    label_to_emotion = {
        index: emotion
        for index, emotion in enumerate(emotion_classes)
    }

    emotion_to_label = {
        emotion: index
        for index, emotion in label_to_emotion.items()
    }

    for label, emotion in label_to_emotion.items():

        print(f"{label} = {emotion}")

    return label_to_emotion, emotion_to_label


# PART 5: VERIFY NUMERICAL LABELS

def verify_labels(
    train_df,
    validation_df,
    test_df,
    config
):

    print("\nVerifying numerical labels...")

    label_column = config["data"]["label_column"]

    num_classes = config["labels"]["num_classes"]

    expected_labels = set(range(num_classes))

    datasets = {
        "Training": train_df,
        "Validation": validation_df,
        "Testing": test_df
    }

    for name, df in datasets.items():

        labels = pd.to_numeric(
            df[label_column],
            errors="coerce"
        )

        if labels.isnull().any():
            raise ValueError(
                f"Invalid labels in {name} dataset."
            )

        if not np.all(labels == np.floor(labels)):
            raise ValueError(
                f"Non-integer labels in {name} dataset."
            )

        actual_labels = set(labels.astype(int).unique())

        if not actual_labels.issubset(expected_labels):
            raise ValueError(
                f"Unexpected labels in {name} dataset."
            )

        print(f"{name} labels verified successfully.")


# PART 6: VERIFY MAPPING

def verify_mapping(
    label_to_emotion,
    emotion_to_label
):

    print("\nVerifying label mapping...")

    for label, emotion in label_to_emotion.items():

        if emotion_to_label[emotion] != label:

            raise ValueError(
                "Emotion label mapping verification failed."
            )

    print("Label mapping verified successfully.")


# PART 7: SAVE LABEL MAPPING

def save_label_mapping(
    label_to_emotion,
    emotion_to_label,
    config
):

    print("\nSaving label mapping...")

    encoder_dir = (
        PROJECT_ROOT
        / config["artifacts"]["encoder_dir"]
    )

    encoder_dir.mkdir(parents=True, exist_ok=True)

    encoder_path = (
        encoder_dir
        / config["artifacts"]["encoder_file"]
    )

    mapping = {
        "label_to_emotion": label_to_emotion,
        "emotion_to_label": emotion_to_label
    }

    with open(encoder_path, "wb") as file:

        pickle.dump(mapping, file)

    print("Label mapping saved successfully.")
    print("Saved location:", encoder_path)


# PART 8: TEST ENCODING AND DECODING

def test_label_mapping(
    label_to_emotion,
    emotion_to_label
):

    print("\nTesting label encoding...")

    sample_emotions = [
        "sadness",
        "joy",
        "love",
        "anger",
        "fear",
        "surprise"
    ]

    encoded = [
        emotion_to_label[emotion]
        for emotion in sample_emotions
    ]

    decoded = [
        label_to_emotion[label]
        for label in encoded
    ]

    print("Original emotions:", sample_emotions)
    print("Encoded labels:", encoded)
    print("Decoded emotions:", decoded)


# PART 9: MAIN FUNCTION

def main():

    print("=" * 55)
    print("EMOTION DETECTION - LABEL ENCODING")
    print("=" * 55)

    config = load_config()

    train_df, validation_df, test_df = load_datasets(config)

    verify_labels(
        train_df,
        validation_df,
        test_df,
        config
    )

    label_to_emotion, emotion_to_label = create_label_mapping(config)

    verify_mapping(
        label_to_emotion,
        emotion_to_label
    )

    save_label_mapping(
        label_to_emotion,
        emotion_to_label,
        config
    )

    test_label_mapping(
        label_to_emotion,
        emotion_to_label
    )

    print("\n" + "=" * 55)
    print("LABEL ENCODING COMPLETED SUCCESSFULLY")
    print("=" * 55)


# PART 10: EXECUTE PROGRAM

if __name__ == "__main__":

    main()
