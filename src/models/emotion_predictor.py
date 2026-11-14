
import re
import pickle
import yaml
import json

from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ==========================================
# PART 1: LOAD CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

CONFIG_PATH = BASE_DIR / "config" / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

print("Configuration loaded successfully.")


# ==========================================
# PART 2: LOAD TRAINED MODEL
# ==========================================

MODEL_PATH = (
    BASE_DIR
    / config["artifacts"]["model_dir"]
    / config["artifacts"]["model_file"]
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Trained model not found. Please execute File 13 first."
    )

model = load_model(MODEL_PATH)

print("Trained LSTM model loaded successfully.")


# ==========================================
# PART 3: LOAD TOKENIZER
# ==========================================

TOKENIZER_PATH = (
    BASE_DIR
    / config["artifacts"]["tokenizer_dir"]
    / config["artifacts"]["tokenizer_file"]
)

if not TOKENIZER_PATH.exists():
    raise FileNotFoundError(
        "Tokenizer not found. Please execute File 7 first."
    )

with open(TOKENIZER_PATH, "rb") as file:
    tokenizer = pickle.load(file)

print("Tokenizer loaded successfully.")


# ==========================================
# PART 4: LOAD LABEL MAPPING
# ==========================================

LABEL_PATH = (
    BASE_DIR
    / config["artifacts"]["encoder_dir"]
    / config["artifacts"]["encoder_file"]
)

if not LABEL_PATH.exists():
    raise FileNotFoundError(
        "Label mapping not found. Please execute File 9 first."
    )

with open(LABEL_PATH, "rb") as file:
    label_mapping = pickle.load(file)

label_to_emotion = label_mapping["label_to_emotion"]

print("Emotion label mapping loaded successfully.")


# ==========================================
# PART 5: TEXT CLEANING FUNCTION
# ==========================================

def clean_text(text):

    text = str(text)

    if config["preprocessing"]["lowercase"]:
        text = text.lower()

    if config["preprocessing"]["remove_urls"]:
        text = re.sub(r"http\S+|www\S+", "", text)

    if config["preprocessing"]["remove_html"]:
        text = re.sub(r"<.*?>", "", text)

    if config["preprocessing"]["remove_extra_spaces"]:
        text = re.sub(r"\s+", " ", text).strip()

    if config["preprocessing"]["remove_special_characters"]:
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    return text


# ==========================================
# PART 6: EMOTION PREDICTION FUNCTION
# ==========================================

def predict_emotion(input_text):

    if not input_text or not input_text.strip():
        raise ValueError("Please enter a valid sentence.")

    max_text_length = config["application"]["max_text_length"]

    if len(input_text) > max_text_length:
        raise ValueError(
            f"Text must not exceed {max_text_length} characters."
        )

    # Clean input sentence
    cleaned_text = clean_text(input_text)

    # Convert text into integer sequence
    sequence = tokenizer.texts_to_sequences([cleaned_text])

    # Apply padding
    padded_sequence = pad_sequences(
        sequence,
        maxlen=config["preprocessing"]["max_sequence_length"],
        padding=config["preprocessing"]["padding_type"],
        truncating=config["preprocessing"]["truncating_type"]
    )

    # Predict emotion probabilities
    probabilities = model.predict(
        padded_sequence,
        verbose=0
    )[0]

    predicted_label = int(probabilities.argmax())

    predicted_emotion = label_to_emotion[predicted_label]

    confidence = float(probabilities[predicted_label]) * 100

    # Prepare all emotion probabilities
    emotion_scores = {
        label_to_emotion[index]: round(
            float(probabilities[index]) * 100, 2
        )
        for index in range(len(probabilities))
    }

    return {
        "input_text": input_text,
        "cleaned_text": cleaned_text,
        "predicted_emotion": predicted_emotion,
        "confidence": round(confidence, 2),
        "emotion_scores": emotion_scores
    }


# ==========================================
# PART 7: DISPLAY PREDICTION
# ==========================================

def display_prediction(result):

    print("\n========== EMOTION PREDICTION ==========")

    print("Input Text:", result["input_text"])

    print("Predicted Emotion:", result["predicted_emotion"])

    print("Confidence:", str(result["confidence"]) + "%")

    print("\nEmotion Probability:")

    for emotion, score in result["emotion_scores"].items():

        print(f"{emotion.capitalize():12}: {score}%")


# ==========================================
# PART 8: USER INPUT
# ==========================================

if __name__ == "__main__":

    print("\n====================================")
    print("       EMOTION DETECTION SYSTEM")
    print("====================================")

    print("Available emotions:")
    print(", ".join(config["labels"]["classes"]))

    while True:

        user_text = input(
            "\nEnter a sentence (or type 'exit' to stop): "
        )

        if user_text.strip().lower() == "exit":
            print("\nEmotion Detection System Closed.")
            break

        try:

            result = predict_emotion(user_text)

            display_prediction(result)

        except ValueError as error:

            print("Input Error:", error)
