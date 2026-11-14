
# ==========================================================
# FILE 10: VOCABULARY BUILDER
# Project: Emotion Detection Using NLP and Deep Learning
# Purpose: Analyze and save vocabulary information
# ==========================================================


# ----------------------------------------------------------
# PART 1: IMPORT LIBRARIES
# ----------------------------------------------------------

import pickle
import yaml
import json

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
# PART 3: LOAD SAVED TOKENIZER
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
            "Tokenizer not found. Execute tokenizer.py first."
        )

    with open(tokenizer_path, "rb") as file:

        tokenizer = pickle.load(file)

    print("Tokenizer loaded successfully.")

    return tokenizer


# ----------------------------------------------------------
# PART 4: BUILD VOCABULARY
# ----------------------------------------------------------

def build_vocabulary(tokenizer, config):

    print("\nBuilding vocabulary...")

    max_words = config["preprocessing"]["max_words"]

    word_index = tokenizer.word_index

    # Keep only words allowed by the tokenizer's vocabulary limit.
    vocabulary = {
        word: index
        for word, index in word_index.items()
        if index < max_words
    }

    print("Vocabulary built successfully.")

    print("Total unique words:", len(word_index))

    print("Usable vocabulary words:", len(vocabulary))

    return vocabulary


# ----------------------------------------------------------
# PART 5: DISPLAY TOP WORDS
# ----------------------------------------------------------

def display_top_words(tokenizer, vocabulary):

    print("\nTop 20 frequent words:")

    word_counts = tokenizer.word_counts

    sorted_words = sorted(
        word_counts.items(),
        key=lambda item: item[1],
        reverse=True
    )

    for word, count in sorted_words[:20]:

        word_id = vocabulary.get(word)

        print(
            f"Word: {word:<20} "
            f"ID: {word_id:<6} "
            f"Frequency: {count}"
        )


# ----------------------------------------------------------
# PART 6: CHECK OOV TOKEN
# ----------------------------------------------------------

def check_oov_token(tokenizer, config):

    print("\nChecking out-of-vocabulary token...")

    oov_token = config["preprocessing"]["oov_token"]

    oov_index = tokenizer.word_index.get(oov_token)

    if oov_index is None:

        raise ValueError(
            "OOV token is missing from the tokenizer."
        )

    print("OOV token:", oov_token)

    print("OOV numerical index:", oov_index)


# ----------------------------------------------------------
# PART 7: SAVE VOCABULARY
# ----------------------------------------------------------

def save_vocabulary(vocabulary, config):

    print("\nSaving vocabulary...")

    vocabulary_dir = PROJECT_ROOT / "artifacts" / "vocabulary"

    vocabulary_dir.mkdir(parents=True, exist_ok=True)

    vocabulary_path = vocabulary_dir / "vocabulary.json"

    with open(
        vocabulary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            vocabulary,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("Vocabulary saved successfully.")

    print("Saved location:", vocabulary_path)


# ----------------------------------------------------------
# PART 8: DISPLAY VOCABULARY STATISTICS
# ----------------------------------------------------------

def display_statistics(vocabulary, tokenizer, config):

    print("\nVocabulary Statistics:")

    print("-" * 45)

    print("Total unique words:", len(tokenizer.word_index))

    print("Usable vocabulary:", len(vocabulary))

    print(
        "Maximum vocabulary setting:",
        config["preprocessing"]["max_words"]
    )

    print(
        "Maximum sequence length:",
        config["preprocessing"]["max_sequence_length"]
    )

    print("Number of emotion classes:", config["labels"]["num_classes"])

    print("-" * 45)


# ----------------------------------------------------------
# PART 9: MAIN FUNCTION
# ----------------------------------------------------------

def main():

    print("=" * 55)

    print("EMOTION DETECTION - VOCABULARY BUILDER")

    print("=" * 55)

    config = load_config()

    tokenizer = load_tokenizer(config)

    vocabulary = build_vocabulary(
        tokenizer,
        config
    )

    display_top_words(
        tokenizer,
        vocabulary
    )

    check_oov_token(
        tokenizer,
        config
    )

    display_statistics(
        vocabulary,
        tokenizer,
        config
    )

    save_vocabulary(
        vocabulary,
        config
    )

    print("\n" + "=" * 55)

    print("VOCABULARY BUILDING COMPLETED SUCCESSFULLY")

    print("=" * 55)


# ----------------------------------------------------------
# PART 10: EXECUTE PROGRAM
# ----------------------------------------------------------

if __name__ == "__main__":

    main()
