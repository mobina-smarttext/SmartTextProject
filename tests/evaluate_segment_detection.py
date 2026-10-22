
import json
import csv
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    brier_score_loss,
    roc_auc_score,
)

from modules.ai_detector.segment_analyzer import (
    analyze_segments
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "Persian": (
        PROJECT_ROOT / "data" / "processed" / "test.jsonl"
    ),
    "English": (
        PROJECT_ROOT / "data" / "english"
        / "processed" / "test.jsonl"
    ),
}

OUTPUT_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_records(path):
    records = []

    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)

            if not isinstance(record, dict):
                raise ValueError(
                    f"Invalid record at line {line_number}"
                )

            records.append(record)

    return records


def evaluate_language(language, dataset_path):

    print("\n" + "=" * 60)
    print(f"Evaluating {language}")
    print("=" * 60)

    records = load_records(dataset_path)

    y_true = []
    y_pred = []
    y_scores = []
    output_rows = []
    skipped = {}

    for index, record in enumerate(records, start=1):

        text = record.get("text")
        label = record.get("label")

        if not isinstance(text, str) or not text.strip():
            reason = "empty_text"
        elif type(label) is not int or label not in (0, 1):
            reason = "invalid_label"
        else:
            reason = None

        if reason is not None:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue

        result = analyze_segments(text)

        segments = result["segments"]

        if len(segments) != 1:
            reason = "not_single_segment"
        elif segments[0]["status"] != "analyzed":
            reason = segments[0]["status"]
        else:
            reason = None

        if reason is not None:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue

        segment = segments[0]

        prediction = int(segment["prediction"])
        ai_score = float(segment["ai_probability"])

        y_true.append(label)
        y_pred.append(prediction)
        y_scores.append(ai_score)

        output_rows.append({
            "record_index": index,
            "true_label": label,
            "predicted_label": prediction,
            "ai_probability": ai_score,
            "word_count": segment["word_count"],
        })

        if index % 100 == 0:
            print(f"Processed {index}/{len(records)}")

    if not y_true:
        print("No valid predictions.")
        return

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1],
            zero_division=0
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    brier = brier_score_loss(y_true, y_scores)

    auc = None
    if len(set(y_true)) == 2:
        auc = roc_auc_score(y_true, y_scores)

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Human", "AI"],
        zero_division=0
    )

    summary = [
        f"Language: {language}",
        f"Total records: {len(records)}",
        f"Evaluated records: {len(y_true)}",
        f"Skipped records: {sum(skipped.values())}",
        f"Skip reasons: {skipped}",
        "",
        f"Accuracy: {accuracy:.4f}",
        f"Brier score: {brier:.4f}",
        f"ROC AUC: {auc:.4f}" if auc is not None
        else "ROC AUC: unavailable",
        "",
        "Per-class metrics:",
        f"Human precision: {precision[0]:.4f}",
        f"Human recall: {recall[0]:.4f}",
        f"Human F1: {f1[0]:.4f}",
        f"Human support: {support[0]}",
        "",
        f"AI precision: {precision[1]:.4f}",
        f"AI recall: {recall[1]:.4f}",
        f"AI F1: {f1[1]:.4f}",
        f"AI support: {support[1]}",
        "",
        "Confusion matrix:",
        str(cm),
        "",
        "Classification report:",
        report,
        "",
        "IMPORTANT:",
        "This is a preliminary single-segment evaluation.",
        "It is not a multi-paragraph document evaluation.",
        "Dataset leakage and source independence must be checked.",
        "Probability calibration on segments needs separate validation.",
    ]

    summary_text = "\n".join(summary)

    print("\n" + summary_text)

    summary_path = (
        OUTPUT_DIR
        / f"segment_evaluation_{language.lower()}.txt"
    )

    summary_path.write_text(
        summary_text,
        encoding="utf-8"
    )

    csv_path = (
        OUTPUT_DIR
        / f"segment_predictions_{language.lower()}.csv"
    )

    with open(
        csv_path,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "record_index",
                "true_label",
                "predicted_label",
                "ai_probability",
                "word_count",
            ]
        )

        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\nSaved report: {summary_path}")
    print(f"Saved predictions: {csv_path}")


if __name__ == "__main__":

    for language, path in DATASETS.items():
        evaluate_language(language, path)
