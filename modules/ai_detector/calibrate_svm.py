
import json
from pathlib import Path

import joblib
import numpy as np

from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LANGUAGES = {
    "persian": {
        "validation": (
            PROJECT_ROOT / "data" / "processed" / "val.jsonl"
        ),
        "model": (
            PROJECT_ROOT / "models" / "ai_detector_svm.joblib"
        ),
        "output": (
            PROJECT_ROOT
            / "models"
            / "ai_detector_svm_calibrated.joblib"
        ),
    },
    "english": {
        "validation": (
            PROJECT_ROOT
            / "data"
            / "english"
            / "processed"
            / "val.jsonl"
        ),
        "model": (
            PROJECT_ROOT
            / "models"
            / "ai_detector_svm_english.joblib"
        ),
        "output": (
            PROJECT_ROOT
            / "models"
            / "ai_detector_svm_english_calibrated.joblib"
        ),
    },
}


def load_dataset(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    texts = []
    labels = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            text = record.get("text", "")
            label = record.get("label")

            if (
                isinstance(text, str)
                and text.strip()
                and type(label) is int
                and label in (0, 1)
            ):
                texts.append(text.strip())
                labels.append(label)

    if not texts:
        raise ValueError(
            f"No valid samples found in: {path}"
        )

    if len(set(labels)) != 2:
        raise ValueError(
            "Calibration data must contain both classes."
        )

    return texts, np.array(labels)


def calibrate_language(language, config):
    print(f"\n{'=' * 55}")
    print(f"Calibrating {language.upper()} model...")
    print(f"{'=' * 55}")

    model_path = config["model"]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    model = joblib.load(model_path)

    X_val, y_val = load_dataset(
        config["validation"]
    )

    print(f"Calibration samples: {len(X_val)}")

    calibrated_model = CalibratedClassifierCV(
        estimator=FrozenEstimator(model),
        method="sigmoid",
    )

    calibrated_model.fit(X_val, y_val)

    probabilities = calibrated_model.predict_proba(
        X_val
    )[:, 1]

    brier = brier_score_loss(
        y_val,
        probabilities
    )

    loss = log_loss(
        y_val,
        calibrated_model.predict_proba(X_val),
        labels=[0, 1]
    )

    print(f"Brier score (calibration data): {brier:.4f}")
    print(f"Log loss (calibration data): {loss:.4f}")

    output_path = config["output"]
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        calibrated_model,
        output_path
    )

    print(f"Calibrated model saved to: {output_path}")


def main():
    for language, config in LANGUAGES.items():
        calibrate_language(
            language,
            config
        )

    print("\nCalibration completed for both languages.")


if __name__ == "__main__":
    main()
