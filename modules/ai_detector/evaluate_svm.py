
import json
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_PATH = PROJECT_DIR / "models" / "ai_detector_svm.joblib"
TEST_PATH = DATA_DIR / "test.jsonl"


def load_dataset(path):
    texts = []
    labels = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)
            text = record.get("text", "")
            label = record.get("label")

            if text.strip() and label in (0, 1):
                texts.append(text)
                labels.append(label)

    return texts, labels


def main():
    print("Loading saved model...")
    model = joblib.load(MODEL_PATH)

    print("Loading test dataset...")
    X_test, y_test = load_dataset(TEST_PATH)

    print(f"Test samples: {len(X_test)}")

    print("\nEvaluating model...")
    predictions = model.predict(X_test)

    print(f"\nAccuracy: {accuracy_score(y_test, predictions):.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Human", "AI-generated"],
            zero_division=0,
        )
    )

    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))


if __name__ == "__main__":
    main()
