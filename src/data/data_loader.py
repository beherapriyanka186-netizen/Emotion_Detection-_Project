
"""
File: src/data/data_loader.py

Purpose:
1. Download the dair-ai Emotion dataset.
2. Load train, validation and test splits.
3. Convert dataset splits into Pandas DataFrames.
4. Save all splits as CSV files.
5. Display dataset information.
"""

from pathlib import Path

import pandas as pd
import yaml
from datasets import load_dataset


# ==========================================
# 1. PROJECT PATH CONFIGURATION
# ==========================================

# Locate the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Configuration file path
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


# ==========================================
# 2. LOAD PROJECT CONFIGURATION
# ==========================================

def load_config():

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_PATH}"
        )

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


# ==========================================
# 3. DOWNLOAD DATASET
# ==========================================

def download_dataset(dataset_name):

    print("\nDownloading Emotion Dataset...")

    dataset = load_dataset(
        dataset_name,
        name="split"
    )

    print("Dataset downloaded successfully!")

    return dataset


# ==========================================
# 4. CONVERT DATASET TO DATAFRAME
# ==========================================

def convert_to_dataframe(dataset):

    print("\nConverting dataset into Pandas DataFrames...")

    train_df = dataset["train"].to_pandas()

    validation_df = dataset["validation"].to_pandas()

    test_df = dataset["test"].to_pandas()

    return train_df, validation_df, test_df


# ==========================================
# 5. SAVE DATASET AS CSV FILES
# ==========================================

def save_dataset(
    train_df,
    validation_df,
    test_df,
    config
):

    data_config = config["data"]

    # Create raw data directory
    raw_dir = PROJECT_ROOT / data_config["raw_dir"]

    raw_dir.mkdir(parents=True, exist_ok=True)

    # Define output file paths
    train_path = PROJECT_ROOT / data_config["train_file"]

    validation_path = (
        PROJECT_ROOT / data_config["validation_file"]
    )

    test_path = PROJECT_ROOT / data_config["test_file"]

    # Create parent directories if necessary
    train_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    # Save CSV files
    train_df.to_csv(train_path, index=False, encoding="utf-8")

    validation_df.to_csv(
        validation_path,
        index=False,
        encoding="utf-8"
    )

    test_df.to_csv(
        test_path,
        index=False,
        encoding="utf-8"
    )

    print("\nDataset CSV files saved successfully!")

    print("Training file:", train_path)
    print("Validation file:", validation_path)
    print("Testing file:", test_path)


# ==========================================
# 6. DISPLAY DATASET INFORMATION
# ==========================================

def display_dataset_info(
    train_df,
    validation_df,
    test_df
):

    print("\n========== DATASET INFORMATION ==========")

    print("\nTraining Dataset:")
    print("Shape:", train_df.shape)

    print("\nValidation Dataset:")
    print("Shape:", validation_df.shape)

    print("\nTesting Dataset:")
    print("Shape:", test_df.shape)

    print("\nEmotion Labels:")

    print(train_df["label"].value_counts().sort_index())

    print("\nSample Training Records:")

    print(train_df.head())

    print("\nTotal Records:")

    total = (
        len(train_df)
        + len(validation_df)
        + len(test_df)
    )

    print("Total:", total)

    print("\n=========================================")


# ==========================================
# 7. MAIN EXECUTION FUNCTION
# ==========================================

def main():

    print("=========================================")
    print("     EMOTION DETECTION PROJECT")
    print("          DATA LOADER")
    print("=========================================")

    # Load configuration
    config = load_config()

    dataset_name = config["data"]["dataset_name"]

    # Download dataset
    dataset = download_dataset(dataset_name)

    # Convert dataset splits
    train_df, validation_df, test_df = (
        convert_to_dataframe(dataset)
    )

    # Validate expected columns
    expected_columns = {
        config["data"]["text_column"],
        config["data"]["label_column"]
    }

    for name, dataframe in [
        ("Training", train_df),
        ("Validation", validation_df),
        ("Testing", test_df)
    ]:

        if not expected_columns.issubset(dataframe.columns):
            raise ValueError(
                f"{name} dataset is missing required columns: "
                f"{expected_columns}"
            )

    # Save dataset
    save_dataset(
        train_df,
        validation_df,
        test_df,
        config
    )

    # Display dataset information
    display_dataset_info(
        train_df,
        validation_df,
        test_df
    )

    print("\nData loading completed successfully!")


# ==========================================
# 8. SCRIPT ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()
