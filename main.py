
from flask import Flask, request, render_template

import yaml
from pathlib import Path

from src.models.emotion_predictor import predict_emotion


# ==========================================
# PART 1: INITIALIZE FLASK APPLICATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "frontend" / "templates"),
    static_folder=str(BASE_DIR / "frontend" / "static")
)

BASE_DIR = Path(__file__).resolve().parent

CONFIG_PATH = BASE_DIR / "config" / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

print("Flask application initialized successfully.")


# ==========================================
# PART 2: HOME PAGE ROUTE
# ==========================================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    error = None
    input_text = ""

    if request.method == "POST":

        input_text = request.form.get("text", "").strip()

        try:

            result = predict_emotion(input_text)

        except ValueError as exception:

            error = str(exception)

        except Exception:

            app.logger.exception("Prediction failed")

            error = (
                "Prediction failed. "
                "Please check the model and try again."
            )

    return render_template(
        "index.html",
        result=result,
        error=error,
        input_text=input_text,
        max_length=config["application"]["max_text_length"]
    )


# ==========================================
# PART 3: RUN FLASK APPLICATION
# ==========================================

if __name__ == "__main__":

    host = config["application"]["host"]
    port = config["application"]["port"]
    debug = config["application"]["debug"]

    print("\n===================================")
    print("     EMOTION DETECTION WEB APP")
    print("===================================")

    print(f"Application URL: http://{host}:{port}")

    app.run(
        host=host,
        port=port,
        debug=debug
    )
