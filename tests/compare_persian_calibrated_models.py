
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OLD_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ai_detector_svm_calibrated.joblib"
)

NEW_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ai_detector_svm_persian_clean_calibrated.joblib"
)

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test.jsonl"
)


def load_test_data():

    texts = []
    labels = []

    with open(
        TEST_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            if not line.strip():
                continue

            record = json.loads(line)

            texts.append(record["text"])
            labels.append(int(record["label"]))

    return texts, np.array(labels)


def get_ai_probabilities(model, texts):

    probabilities = model.predict_proba(texts)

    classes = list(model.classes_)

    return probabilities[:, classes.index(1)]


def main():

    print("\n" + "=" * 65)
    print("PERSIAN CALIBRATED MODEL COMPARISON")
    print("=" * 65)

    print("\nLoading models...")

    old_model = joblib.load(OLD_MODEL_PATH)
    new_model = joblib.load(NEW_MODEL_PATH)

    texts, labels = load_test_data()

    print(f"Test samples: {len(texts)}")

    print("\nGenerating predictions...")

    old_predictions = old_model.predict(texts)
    new_predictions = new_model.predict(texts)

    old_probabilities = get_ai_probabilities(
        old_model,
        texts
    )

    new_probabilities = get_ai_probabilities(
        new_model,
        texts
    )

    differences = np.abs(
        old_probabilities - new_probabilities
    )

    changed_predictions = int(
        np.sum(old_predictions != new_predictions)
    )

    old_accuracy = accuracy_score(
        labels,
        old_predictions
    )

    new_accuracy = accuracy_score(
        labels,
        new_predictions
    )

    print("\nResults:")
    print("-" * 45)

    print(
        f"Old model accuracy: {old_accuracy:.6f}"
    )

    print(
        f"New model accuracy: {new_accuracy:.6f}"
    )

    print(
        "Changed predicted labels:",
        changed_predictions
    )

    print(
        "Mean absolute probability difference:",
        f"{np.mean(differences):.8f}"
    )

    print(
        "Maximum probability difference:",
        f"{np.max(differences):.8f}"
    )

    print(
        "Median probability difference:",
        f"{np.median(differences):.8f}"
    )

    print(
        "Samples with probability difference > 0.01:",
        int(np.sum(differences > 0.01))
    )

    print(
        "Samples with probability difference > 0.05:",
        int(np.sum(differences > 0.05))
    )

    print("\nComparison completed.")


if __name__ == "__main__":
    main()
