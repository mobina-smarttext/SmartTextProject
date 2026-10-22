
import json
from pathlib import Path

import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "english"
    / "processed"
    / "test.jsonl"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ai_detector_svm_english.joblib"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "svm_english_evaluation_report.txt"
)


def load_test_data():
    texts = []
    labels = []

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_FILE}"
        )

    with TEST_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
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
        raise ValueError("Test dataset is empty.")

    return texts, labels


def main():
    print("Loading English SVM model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    print("Loading independent English test dataset...")

    X_test, y_test = load_test_data()

    print(f"Test samples: {len(X_test)}")

    print("\nEvaluating model...")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    classification = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Human", "AI-generated"],
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    report = (
        "ENGLISH AI DETECTOR - TEST EVALUATION\n"
        + "=" * 55
        + f"\nTest samples: {len(X_test)}"
        + f"\nAccuracy: {accuracy:.4f}"
        + f"\nPrecision (AI): {precision:.4f}"
        + f"\nRecall (AI): {recall:.4f}"
        + f"\nF1-score (AI): {f1:.4f}"
        + f"\nMacro F1-score: {macro_f1:.4f}"
        + "\n\nClassification report:\n"
        + classification
        + "\nConfusion matrix "
        + "(rows=true, columns=predicted):\n"
        + str(matrix)
        + "\n\nNote: These results apply to the HC3 "
        + "test split and do not guarantee "
        + "performance on other datasets."
    )

    print("\n" + report)

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_PATH.write_text(
        report,
        encoding="utf-8"
    )

    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
