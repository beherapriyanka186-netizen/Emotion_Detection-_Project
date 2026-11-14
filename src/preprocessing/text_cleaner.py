
# ==========================================================
# FILE 6: TEXT CLEANER
# Project: Emotion Detection Using NLP and Deep Learning
# Purpose: Clean text data before model training
# ==========================================================


# ----------------------------------------------------------
# PART 1: IMPORT LIBRARIES
# ----------------------------------------------------------

import re
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
# PART 3: TEXT CLEANING FUNCTION
# ----------------------------------------------------------

def clean_text(text, config):

    if pd.isna(text):
        return ""

    text = str(text)

    preprocessing = config["preprocessing"]

    # Convert text into lowercase
    if preprocessing["lowercase"]:
        text = text.lower()

    # Remove URLs
    if preprocessing["remove_urls"]:
        text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove HTML tags
    if preprocessing["remove_html"]:
        text = re.sub(r"<.*?>", "", text)

    # Remove extra spaces
    if preprocessing["remove_extra_spaces"]:
        text = re.sub(r"\s+", " ", text).strip()

    # Remove special characters if enabled
    if preprocessing["remove_special_characters"]:
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    return text


# ----------------------------------------------------------
# PART 4: LOAD PROCESSED DATASETS
# ----------------------------------------------------------

def load_datasets(config):

    print("\nLoading datasets...")

    data_dir = PROJECT_ROOT / config["data"]["processed_dir"]

    train_df = pd.read_csv(data_dir / "train.csv")
    validation_df = pd.read_csv(data_dir / "validation.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    print("Training records:", len(train_df))
    print("Validation records:", len(validation_df))
    print("Testing records:", len(test_df))

    return train_df, validation_df, test_df


# ----------------------------------------------------------
# PART 5: APPLY TEXT CLEANING
# ----------------------------------------------------------

def preprocess_dataset(df, config):

    text_column = config["data"]["text_column"]

    cleaned_df = df.copy()

    cleaned_df[text_column] = cleaned_df[text_column].apply(
        lambda text: clean_text(text, config)
    )

    return cleaned_df


# ----------------------------------------------------------
# PART 6: PROCESS ALL DATASETS
# ----------------------------------------------------------

def process_all_datasets(train_df, validation_df, test_df, config):

    print("\nCleaning training dataset...")
    train_cleaned = preprocess_dataset(train_df, config)

    print("Cleaning validation dataset...")
    validation_cleaned = preprocess_dataset(validation_df, config)

    print("Cleaning testing dataset...")
    test_cleaned = preprocess_dataset(test_df, config)

    return train_cleaned, validation_cleaned, test_cleaned


# ----------------------------------------------------------
# PART 7: DISPLAY CLEANED TEXT
# ----------------------------------------------------------

def display_samples(train_df, config):

    text_column = config["data"]["text_column"]

    print("\nOriginal and cleaned text samples:")

    original_dir = PROJECT_ROOT / config["data"]["processed_dir"]

    original_df = pd.read_csv(original_dir / "train.csv")

    for index in range(min(5, len(train_df))):

        print("\nSample:", index + 1)

        print("Original Text:")
        print(original_df[text_column].iloc[index])

        print("Cleaned Text:")
        print(train_df[text_column].iloc[index])


# ----------------------------------------------------------
# PART 8: SAVE CLEANED DATASETS
# ----------------------------------------------------------

def save_cleaned_datasets(
    train_df,
    validation_df,
    test_df,
    config
):

    print("\nSaving cleaned datasets...")

    output_dir = (
        PROJECT_ROOT
        / config["data"]["processed_dir"]
        / "cleaned"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(
        output_dir / "train_cleaned.csv",
        index=False
    )

    validation_df.to_csv(
        output_dir / "validation_cleaned.csv",
        index=False
    )

    test_df.to_csv(
        output_dir / "test_cleaned.csv",
        index=False
    )

    print("Cleaned training dataset saved.")
    print("Cleaned validation dataset saved.")
    print("Cleaned testing dataset saved.")

    print("Output directory:", output_dir)


# ----------------------------------------------------------
# PART 9: MAIN FUNCTION
# ----------------------------------------------------------

def main():

    print("=" * 55)
    print("EMOTION DETECTION - TEXT CLEANING")
    print("=" * 55)

    config = load_config()

    train_df, validation_df, test_df = load_datasets(config)

    (
        train_cleaned,
        validation_cleaned,
        test_cleaned
    ) = process_all_datasets(
        train_df,
        validation_df,
        test_df,
        config
    )

    display_samples(train_cleaned, config)

    save_cleaned_datasets(
        train_cleaned,
        validation_cleaned,
        test_cleaned,
        config
    )

    print("\n" + "=" * 55)
    print("TEXT CLEANING COMPLETED SUCCESSFULLY")
    print("=" * 55)


# ----------------------------------------------------------
# PART 10: EXECUTE PROGRAM
# ----------------------------------------------------------

if __name__ == "__main__":
    main()
