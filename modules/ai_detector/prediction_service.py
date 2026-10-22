
from pathlib import Path

import joblib

from modules.ai_detector.language_detector import detect_language


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATHS = {
    "Persian": (
        PROJECT_ROOT
        / "models"
        / "ai_detector_svm_calibrated.joblib"
    ),
    "English": (
        PROJECT_ROOT
        / "models"
        / "ai_detector_svm_english_calibrated.joblib"
    ),
}


def load_calibrated_model(language):
    if language not in MODEL_PATHS:
        raise ValueError(
            f"Unsupported language: {language}"
        )

    path = MODEL_PATHS[language]

    if not path.exists():
        raise FileNotFoundError(
            f"Calibrated model not found: {path}"
        )

    return joblib.load(path)


def predict_ai_probability(text, model=None):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text must not be empty.")

    language, mixed_language = detect_language(text)

    if language == "Unknown":
        raise ValueError(
            "Could not detect Persian or English text."
        )

    if model is None:
        model = load_calibrated_model(language)

    probabilities = model.predict_proba([text.strip()])[0]

    classes = list(model.classes_)

    if 0 not in classes or 1 not in classes:
        raise ValueError(
            f"Unexpected model classes: {classes}"
        )

    ai_probability = float(
        probabilities[classes.index(1)]
    )

    human_probability = float(
        probabilities[classes.index(0)]
    )

    prediction = int(model.predict([text.strip()])[0])

    return {
        "language": language,
        "mixed_language": mixed_language,
        "prediction": prediction,
        "ai_probability": ai_probability,
        "human_probability": human_probability,
    }
