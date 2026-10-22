
import json
from pathlib import Path

import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_DIR / "data" / "english" / "processed"

MODEL_DIR = PROJECT_DIR / "models"

MODEL_PATH = MODEL_DIR / "ai_detector_svm_english.joblib"


def load_dataset(filename):
    texts = []
    labels = []

    file_path = DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}"
        )

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)

            text = record.get("text", "")
            label = record.get("label")

            if (
                isinstance(text, str)
                and text.strip()
                and label in (0, 1)
            ):
                texts.append(text.strip())
                labels.append(label)

    return texts, labels


def main():

    print("Loading English datasets...")

    X_train, y_train = load_dataset("train.jsonl")
    X_val, y_val = load_dataset("val.jsonl")

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    if not X_train or not X_val:
        raise ValueError(
            "Training and validation datasets must not be empty."
        )

    if len(set(y_train)) < 2:
        raise ValueError(
            "Training dataset must contain both Human and AI labels."
        )

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
            LinearSVC(
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    print("\nTraining English SVM model...")

    model.fit(X_train, y_train)

    print("\nValidation results:")

    predictions = model.predict(X_val)

    print(
        f"Accuracy: "
        f"{accuracy_score(y_val, predictions):.4f}"
    )

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

    print(
        "Confusion matrix "
        "(rows=true, columns=predicted):"
    )

    print(
        confusion_matrix(
            y_val,
            predictions,
            labels=[0, 1]
        )
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        f"\nEnglish model saved to: {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()
