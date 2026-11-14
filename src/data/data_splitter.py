
# ==========================================================
# FILE 5: DATA SPLITTER
# Project: Emotion Detection Using NLP and Deep Learning
# Purpose: Prepare and verify train, validation and test data
# ==========================================================


# ----------------------------------------------------------
# PART 1: IMPORT LIBRARIES
# ----------------------------------------------------------

import pandas as pd
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
# PART 3: LOAD ORIGINAL DATASETS
# ----------------------------------------------------------

def load_datasets(config):

    print("\nLoading original datasets...")

    train_path = PROJECT_ROOT / config["data"]["train_file"]
    validation_path = PROJECT_ROOT / config["data"]["validation_file"]
    test_path = PROJECT_ROOT / config["data"]["test_file"]

    train_df = pd.read_csv(train_path)
    validation_df = pd.read_csv(validation_path)
    test_df = pd.read_csv(test_path)

    print("Training dataset loaded:", train_df.shape)
    print("Validation dataset loaded:", validation_df.shape)
    print("Testing dataset loaded:", test_df.shape)

    return train_df, validation_df, test_df


# ----------------------------------------------------------
# PART 4: VERIFY DATASETS
# ----------------------------------------------------------

def verify_datasets(train_df, validation_df, test_df, config):

    print("\nVerifying datasets...")

    text_column = config["data"]["text_column"]
    label_column = config["data"]["label_column"]

    datasets = {
        "Training": train_df,
        "Validation": validation_df,
        "Testing": test_df
    }

    for name, df in datasets.items():

        if df.empty:
            raise ValueError(f"{name} dataset is empty.")

        if text_column not in df.columns:
            raise ValueError(f"Text column missing in {name} dataset.")

        if label_column not in df.columns:
            raise ValueError(f"Label column missing in {name} dataset.")

        if df[text_column].isnull().any():
            raise ValueError(f"Missing text values found in {name} dataset.")

        if df[label_column].isnull().any():
            raise ValueError(f"Missing labels found in {name} dataset.")

        print(f"{name} dataset verification successful.")

    print("\nAll datasets verified successfully.")


# ----------------------------------------------------------
# PART 5: CHECK DATA LEAKAGE
# ----------------------------------------------------------

def check_data_leakage(train_df, validation_df, test_df, config):

    print("\nChecking duplicate text across datasets...")

    text_column = config["data"]["text_column"]

    train_texts = set(train_df[text_column].astype(str))
    validation_texts = set(validation_df[text_column].astype(str))
    test_texts = set(test_df[text_column].astype(str))

    train_validation = train_texts.intersection(validation_texts)
    train_test = train_texts.intersection(test_texts)
    validation_test = validation_texts.intersection(test_texts)

    print("Train-Validation overlap:", len(train_validation))
    print("Train-Test overlap:", len(train_test))
    print("Validation-Test overlap:", len(validation_test))

    if train_validation or train_test or validation_test:

        print(
            "\nWARNING: Duplicate text records exist across "
            "one or more datasets."
        )

        print(
            "The original splits will be preserved. "
            "Review these records before model training."
        )

    else:

        print("No duplicate text overlap detected.")


# ----------------------------------------------------------
# PART 6: DISPLAY CLASS DISTRIBUTION
# ----------------------------------------------------------

def display_class_distribution(
    train_df,
    validation_df,
    test_df,
    config
):

    print("\nEmotion class distribution:")

    label_column = config["data"]["label_column"]

    datasets = {
        "Training": train_df,
        "Validation": validation_df,
        "Testing": test_df
    }

    for name, df in datasets.items():

        print(f"\n{name} Dataset:")

        distribution = df[label_column].value_counts().sort_index()

        print(distribution)


# ----------------------------------------------------------
# PART 7: SAVE PROCESSED DATASETS
# ----------------------------------------------------------

def save_datasets(
    train_df,
    validation_df,
    test_df,
    config
):

    print("\nSaving prepared datasets...")

    output_dir = PROJECT_ROOT / config["data"]["processed_dir"]

    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(
        output_dir / "train.csv",
        index=False
    )

    validation_df.to_csv(
        output_dir / "validation.csv",
        index=False
    )

    test_df.to_csv(
        output_dir / "test.csv",
        index=False
    )

    print("Training data saved.")
    print("Validation data saved.")
    print("Testing data saved.")

    print("\nProcessed data directory:", output_dir)


# ----------------------------------------------------------
# PART 8: MAIN FUNCTION
# ----------------------------------------------------------

def main():

    print("=" * 55)
    print("EMOTION DETECTION - DATA SPLITTING PROCESS")
    print("=" * 55)

    config = load_config()

    train_df, validation_df, test_df = load_datasets(config)

    verify_datasets(
        train_df,
        validation_df,
        test_df,
        config
    )

    check_data_leakage(
        train_df,
        validation_df,
        test_df,
        config
    )

    display_class_distribution(
        train_df,
        validation_df,
        test_df,
        config
    )

    save_datasets(
        train_df,
        validation_df,
        test_df,
        config
    )

    print("\n" + "=" * 55)
    print("DATA PREPARATION COMPLETED SUCCESSFULLY")
    print("=" * 55)


# ----------------------------------------------------------
# PART 9: EXECUTE PROGRAM
# ----------------------------------------------------------

if __name__ == "__main__":
    main()
