
import json
from pathlib import Path

import joblib
from sklearn.calibration import (
    CalibratedClassifierCV,
    FrozenEstimator,
)
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ai_detector_svm.joblib"
)

CLEAN_VAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "cleaned_experimental"
    / "persian"
    / "val.jsonl"
)

CLEAN_TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "cleaned_experimental"
    / "persian"
    / "test.jsonl"
)

OUTPUT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ai_detector_svm_persian_clean_calibrated.joblib"
)

OUTPUT_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "persian_clean_calibration_report.txt"
)


def load_dataset(path):

    texts = []
    labels = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            if not line.strip():
                continue

            record = json.loads(line)

            text = record["text"]
            label = int(record["label"])

            texts.append(text)
            labels.append(label)

    return texts, labels


def main():

    print("\n" + "=" * 65)
    print("PERSIAN CLEAN VALIDATION RECALIBRATION")
    print("=" * 65)

    print("\nLoading original SVM model...")

    base_model = joblib.load(
        BASE_MODEL_PATH
    )

    print("Loading cleaned validation data...")

    val_texts, val_labels = load_dataset(
        CLEAN_VAL_PATH
    )

    print(
        f"Validation samples: {len(val_texts)}"
    )

    print("\nCalibrating probabilities...")

    calibrated_model = CalibratedClassifierCV(
        estimator=FrozenEstimator(base_model),
        method="sigmoid"
    )

    calibrated_model.fit(
        val_texts,
        val_labels
    )

    print("Calibration completed.")

    print("\nEvaluating on unchanged test data...")

    test_texts, test_labels = load_dataset(
        CLEAN_TEST_PATH
    )

    predictions = calibrated_model.predict(
        test_texts
    )

    probabilities = (
        calibrated_model.predict_proba(
            test_texts
        )
    )

    classes = list(
        calibrated_model.classes_
    )

    ai_index = classes.index(1)

    ai_probabilities = (
        probabilities[:, ai_index]
    )

    accuracy = accuracy_score(
        test_labels,
        predictions
    )

    brier = brier_score_loss(
        test_labels,
        ai_probabilities
    )

    auc = roc_auc_score(
        test_labels,
        ai_probabilities
    )

    report = "\n".join([
        "Persian Clean Calibration Evaluation",
        "=" * 45,
        "",
        f"Validation samples: {len(val_texts)}",
        f"Test samples: {len(test_texts)}",
        "",
        f"Accuracy: {accuracy:.4f}",
        f"Brier score: {brier:.4f}",
        f"ROC AUC: {auc:.4f}",
        "",
        "Base SVM: original unchanged model",
        "Calibration: sigmoid",
        "Validation: cleaned experimental split",
        "Test: original unchanged test split",
        "",
        "Note:",
        "This is a recalibration experiment.",
        "The original SVM was not retrained.",
        "The original calibrated model was not overwritten.",
    ])

    OUTPUT_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        calibrated_model,
        OUTPUT_MODEL_PATH
    )

    OUTPUT_REPORT_PATH.write_text(
        report,
        encoding="utf-8"
    )

    print("\n" + report)

    print(
        "\nSaved new model:",
        OUTPUT_MODEL_PATH
    )

    print(
        "Saved report:",
        OUTPUT_REPORT_PATH
    )

    print(
        "\nOriginal models: UNCHANGED"
    )

    print(
        "Recalibration experiment completed."
    )


if __name__ == "__main__":
    main()
