
import json
from pathlib import Path

import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIGS = {
    "Persian": {
        "test_file": (
            PROJECT_ROOT / "data" / "processed" / "test.jsonl"
        ),
        "model_file": (
            PROJECT_ROOT
            / "models"
            / "ai_detector_svm_calibrated.joblib"
        ),
    },
    "English": {
        "test_file": (
            PROJECT_ROOT
            / "data"
            / "english"
            / "processed"
            / "test.jsonl"
        ),
        "model_file": (
            PROJECT_ROOT
            / "models"
            / "ai_detector_svm_english_calibrated.joblib"
        ),
    },
}

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "calibrated_svm_evaluation_report.txt"
)


def load_test_data(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Test file not found: {path}"
        )

    texts = []
    labels = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)

            text = record.get("text", "")
            label = record.get("label")

            if not (
                isinstance(text, str)
                and text.strip()
                and type(label) is int
                and label in (0, 1)
            ):
                raise ValueError(
                    f"Invalid record at line {line_number}: {path}"
                )

            texts.append(text.strip())
            labels.append(label)

    if not texts:
        raise ValueError(f"Empty test dataset: {path}")

    if len(set(labels)) != 2:
        raise ValueError(
            f"Both classes are required: {path}"
        )

    return texts, np.asarray(labels, dtype=int)


def evaluate_language(language, config):
    print("\n" + "=" * 60)
    print(f"Evaluating calibrated {language} SVM")
    print("=" * 60)

    model_path = config["model_file"]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Calibrated model not found: {model_path}"
        )

    model = joblib.load(model_path)

    X_test, y_test = load_test_data(
        config["test_file"]
    )

    probabilities = model.predict_proba(X_test)

    classes = list(model.classes_)

    if 0 not in classes or 1 not in classes:
        raise ValueError(
            f"Unexpected model classes: {classes}"
        )

    ai_index = classes.index(1)

    ai_probabilities = probabilities[:, ai_index]

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    brier = brier_score_loss(
        y_test,
        ai_probabilities
    )

    loss = log_loss(
        y_test,
        probabilities,
        labels=classes
    )

    auc = roc_auc_score(
        y_test,
        ai_probabilities
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

    result = (
        f"\n{'=' * 60}\n"
        f"{language.upper()} CALIBRATED SVM - TEST RESULTS\n"
        f"{'=' * 60}\n"
        f"Test samples: {len(X_test)}\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Brier Score: {brier:.4f}\n"
        f"Log Loss: {loss:.4f}\n"
        f"ROC AUC: {auc:.4f}\n\n"
        f"Classification report:\n{classification}\n"
        f"Confusion matrix "
        f"(rows=true, columns=predicted):\n{matrix}\n"
    )

    print(result)

    return result


def main():
    reports = []

    for language, config in CONFIGS.items():
        report = evaluate_language(
            language,
            config
        )
        reports.append(report)

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_PATH.write_text(
        "\n".join(reports)
        + "\n\nNote: Results are specific to the "
        + "test datasets used. They do not prove "
        + "that individual predictions are certain.",
        encoding="utf-8"
    )

    print("\nAll evaluations completed.")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
