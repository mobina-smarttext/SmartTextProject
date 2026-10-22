
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "ai_detector_svm.joblib"


def load_dataset(filename):
    texts = []
    labels = []

    file_path = DATA_DIR / filename

    with file_path.open("r", encoding="utf-8") as file:
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
    print("Loading datasets...")

    X_train, y_train = load_dataset("train.jsonl")
    X_val, y_val = load_dataset("val.jsonl")

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=100000,
                sublinear_tf=True,
            ),
        ),
        (
            "svm",
            LinearSVC(class_weight="balanced", random_state=42),
        ),
    ])

    print("\nTraining SVM model...")
    model.fit(X_train, y_train)

    print("\nValidation results:")
    predictions = model.predict(X_val)

    print(f"Accuracy: {accuracy_score(y_val, predictions):.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_val,
            predictions,
            labels=[0, 1],
            target_names=["Human", "AI-generated"],
            zero_division=0,
        )
    )

    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_val, predictions, labels=[0, 1]))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    import joblib
    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
