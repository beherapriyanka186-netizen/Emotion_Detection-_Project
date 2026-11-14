
# ==========================================================
# FILE 7: TOKENIZER
# Project: Emotion Detection Using NLP and Deep Learning
# Purpose: Convert cleaned text into numerical sequences
# ==========================================================


# ----------------------------------------------------------
# PART 1: IMPORT LIBRARIES
# ----------------------------------------------------------

import pandas as pd
import yaml
import pickle

from pathlib import Path
from tensorflow.keras.preprocessing.text import Tokenizer


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
# PART 4: CREATE TOKENIZER
# ----------------------------------------------------------

def create_tokenizer(train_df, config):

    print("\nCreating tokenizer...")

    text_column = config["data"]["text_column"]

    max_words = config["preprocessing"]["max_words"]

    oov_token = config["preprocessing"]["oov_token"]

    tokenizer = Tokenizer(
        num_words=max_words,
        oov_token=oov_token
    )

    # Learn vocabulary from training text only
    tokenizer.fit_on_texts(
        train_df[text_column].astype(str).tolist()
    )

    print("Tokenizer created successfully.")

    print("Vocabulary size:", len(tokenizer.word_index))

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

    print("Training sequences created.")
    print("Validation sequences created.")
    print("Testing sequences created.")

    return (
        train_sequences,
        validation_sequences,
        test_sequences
    )


# ----------------------------------------------------------
# PART 6: SAVE TOKENIZER
# ----------------------------------------------------------

def save_tokenizer(tokenizer, config):

    print("\nSaving tokenizer...")

    tokenizer_dir = (
        PROJECT_ROOT
        / config["artifacts"]["tokenizer_dir"]
    )

    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = (
        tokenizer_dir
        / config["artifacts"]["tokenizer_file"]
    )

    with open(tokenizer_path, "wb") as file:

        pickle.dump(tokenizer, file)

    print("Tokenizer saved successfully.")

    print("Tokenizer path:", tokenizer_path)


# ----------------------------------------------------------
# PART 7: DISPLAY TOKENIZATION SAMPLES
# ----------------------------------------------------------

def display_samples(train_df, train_sequences, config):

    text_column = config["data"]["text_column"]

    print("\nTokenization examples:")

    for index in range(min(5, len(train_df))):

        print("\nSample:", index + 1)

        print("Original Text:")
        print(train_df[text_column].iloc[index])

        print("Numerical Sequence:")
        print(train_sequences[index])


# ----------------------------------------------------------
# PART 8: MAIN FUNCTION
# ----------------------------------------------------------

def main():

    print("=" * 55)

    print("EMOTION DETECTION - TOKENIZATION")

    print("=" * 55)

    config = load_config()

    (
        train_df,
        validation_df,
        test_df
    ) = load_cleaned_datasets(config)

    tokenizer = create_tokenizer(
        train_df,
        config
    )

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

    display_samples(
        train_df,
        train_sequences,
        config
    )

    save_tokenizer(
        tokenizer,
        config
    )

    print("\nSequence counts:")

    print("Training:", len(train_sequences))
    print("Validation:", len(validation_sequences))
    print("Testing:", len(test_sequences))

    print("\n" + "=" * 55)

    print("TOKENIZATION COMPLETED SUCCESSFULLY")

    print("=" * 55)


# ----------------------------------------------------------
# PART 9: EXECUTE PROGRAM
# ----------------------------------------------------------

if __name__ == "__main__":

    main()
