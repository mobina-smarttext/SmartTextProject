
from pathlib import Path
from datetime import datetime
import json

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT / "models" / "ai_detector_svm.joblib"
)

TEST_PATH = (
    PROJECT_ROOT / "data" / "processed" / "test.jsonl"
)

REPORTS_FOLDER = PROJECT_ROOT / "reports"
REPORTS_FOLDER.mkdir(parents=True, exist_ok=True)


def load_test_data():
    """خواندن داده‌های آزمون."""

    texts = []
    labels = []

    with TEST_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            texts.append(record["text"])
            labels.append(int(record["label"]))

    return texts, labels


def main():
    print("Loading saved SVM model...")
    model = joblib.load(MODEL_PATH)

    print("Loading test dataset...")
    texts, true_labels = load_test_data()

    print("Evaluating model...")
    predicted_labels = model.predict(texts)

    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=["Human", "AI-generated"],
        digits=4,
        zero_division=0
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=[0, 1]
    )

    report_text = (
        "SVM AI-Generated Text Detection - Evaluation Report\n"
        + "=" * 60
        + "\n\nEvaluation date: "
        + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        + "\n\nDataset: Persian AI Text Corpus"
        + "\nTest samples: "
        + str(len(texts))
        + "\nHuman label: 0"
        + "\nAI-generated label: 1"
        + "\n\nAccuracy: "
        + f"{accuracy:.4f}"
        + "\n\nClassification Report:\n"
        + report
        + "\nConfusion Matrix (rows=true, columns=predicted):\n"
        + str(matrix)
        + "\n\nLimitations:\n"
        + "- Results are based on the current Persian test dataset.\n"
        + "- Performance on other domains or languages is not established.\n"
        + "- Predictions do not prove whether a text was written by AI.\n"
    )

    output_file = (
        REPORTS_FOLDER / "svm_evaluation_report.txt"
    )

    output_file.write_text(
        report_text,
        encoding="utf-8-sig"
    )

    print("\n" + report_text)
    print("\nReport saved successfully!")
    print(f"File: {output_file}")


if __name__ == "__main__":
    main()
