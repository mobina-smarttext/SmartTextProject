
import csv
import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    brier_score_loss,
    roc_auc_score,
)

from modules.ai_detector.segment_analyzer import (
    analyze_segments,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "multiparagraph"
)

REPORT_DIR = PROJECT_ROOT / "reports"

DATASETS = {
    "Persian": DATA_DIR / "persian_multiparagraph.jsonl",
    "English": DATA_DIR / "english_multiparagraph.jsonl",
}


def load_documents(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    documents = []

    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            document = json.loads(line)

            if not isinstance(document, dict):
                raise ValueError(
                    f"Invalid document at line {line_number}"
                )

            documents.append(document)

    return documents


def evaluate_language(language, path):

    print("\n" + "=" * 65)
    print(f"MULTIPARAGRAPH EVALUATION: {language}")
    print("=" * 65, flush=True)

    documents = load_documents(path)

    true_labels = []
    predicted_labels = []
    ai_probabilities = []

    skipped_reasons = Counter()
    output_rows = []

    total_segments = 0
    fully_analyzed_documents = 0

    for doc_index, document in enumerate(
        documents, start=1
    ):

        document_id = document["document_id"]
        text = document["text"]
        expected_segments = document["segments"]

        # بررسی سازگاری متن سند با برچسب‌ها
        reconstructed_text = "\n\n".join(
            item["text"]
            for item in expected_segments
        )

        if text != reconstructed_text:
            raise ValueError(
                f"Document text mismatch: {document_id}"
            )

        result = analyze_segments(text)

        predicted_segments = result["segments"]

        if len(predicted_segments) != len(expected_segments):
            raise ValueError(
                f"Segment count mismatch in {document_id}: "
                f"expected {len(expected_segments)}, "
                f"got {len(predicted_segments)}"
            )

        document_analyzed = 0

        for expected, predicted in zip(
            expected_segments,
            predicted_segments
        ):

            segment_number = expected["segment_number"]
            true_label = expected["label"]

            if expected["text"] != predicted["text"]:
                raise ValueError(
                    f"Segment text mismatch: "
                    f"{document_id}, segment {segment_number}"
                )

            if type(true_label) is not int or true_label not in (0, 1):
                raise ValueError(
                    f"Invalid label in {document_id}"
                )

            total_segments += 1

            status = predicted["status"]

            row = {
                "document_id": document_id,
                "language": language,
                "segment_number": segment_number,
                "group_id": expected["group_id"],
                "true_label": true_label,
                "status": status,
                "predicted_label": "",
                "ai_probability": "",
                "word_count": predicted["word_count"],
            }

            if status == "analyzed":

                prediction = int(
                    predicted["prediction"]
                )

                probability = float(
                    predicted["ai_probability"]
                )

                true_labels.append(true_label)
                predicted_labels.append(prediction)
                ai_probabilities.append(probability)

                row["predicted_label"] = prediction
                row["ai_probability"] = probability

                document_analyzed += 1

            else:
                skipped_reasons[status] += 1

            output_rows.append(row)

        if document_analyzed == len(expected_segments):
            fully_analyzed_documents += 1

        if doc_index % 10 == 0:
            print(
                f"Processed {doc_index}/{len(documents)} documents",
                flush=True
            )

    evaluated_count = len(true_labels)
    skipped_count = total_segments - evaluated_count

    coverage = (
        evaluated_count / total_segments
        if total_segments else 0
    )

    summary = [
        f"Language: {language}",
        f"Documents: {len(documents)}",
        f"Total segments: {total_segments}",
        f"Evaluated segments: {evaluated_count}",
        f"Skipped segments: {skipped_count}",
        f"Coverage: {coverage:.4f}",
        f"Skip reasons: {dict(skipped_reasons)}",
        f"Fully analyzed documents: "
        f"{fully_analyzed_documents}",
        "",
    ]

    if evaluated_count > 0:

        accuracy = accuracy_score(
            true_labels,
            predicted_labels
        )

        precision, recall, f1, support = (
            precision_recall_fscore_support(
                true_labels,
                predicted_labels,
                labels=[0, 1],
                zero_division=0
            )
        )

        cm = confusion_matrix(
            true_labels,
            predicted_labels,
            labels=[0, 1]
        )

        report = classification_report(
            true_labels,
            predicted_labels,
            labels=[0, 1],
            target_names=["Human", "AI"],
            zero_division=0
        )

        summary.extend([
            f"Accuracy: {accuracy:.4f}",
            "",
            "Human metrics:",
            f"Precision: {precision[0]:.4f}",
            f"Recall: {recall[0]:.4f}",
            f"F1: {f1[0]:.4f}",
            f"Support: {support[0]}",
            "",
            "AI metrics:",
            f"Precision: {precision[1]:.4f}",
            f"Recall: {recall[1]:.4f}",
            f"F1: {f1[1]:.4f}",
            f"Support: {support[1]}",
            "",
            "Confusion matrix:",
            str(cm),
            "",
            "Classification report:",
            report,
        ])

        brier = brier_score_loss(
            true_labels,
            ai_probabilities
        )

        summary.append(
            f"Brier score: {brier:.4f}"
        )

        if len(set(true_labels)) == 2:

            auc = roc_auc_score(
                true_labels,
                ai_probabilities
            )

            summary.append(
                f"ROC AUC: {auc:.4f}"
            )

    summary.extend([
        "",
        "IMPORTANT LIMITATIONS:",
        "Documents are synthetically assembled from "
        "previously used held-out test samples.",
        "This is not an independent external test.",
        "Metrics exclude skipped segments.",
        "Coverage must be reported alongside accuracy.",
        "Paragraphs may be unrelated in topic.",
        "The currently configured production models "
        "are used by analyze_segments.",
        "Results do not establish performance on "
        "authentic student documents.",
    ])

    summary_text = "\n".join(summary)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report_path = (
        REPORT_DIR
        / f"multiparagraph_evaluation_{language.lower()}.txt"
    )

    report_path.write_text(
        summary_text,
        encoding="utf-8"
    )

    csv_path = (
        REPORT_DIR
        / f"multiparagraph_predictions_{language.lower()}.csv"
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
                "document_id",
                "language",
                "segment_number",
                "group_id",
                "true_label",
                "status",
                "predicted_label",
                "ai_probability",
                "word_count",
            ]
        )

        writer.writeheader()
        writer.writerows(output_rows)

    print("\n" + summary_text)

    print(f"\nSaved report: {report_path}")
    print(f"Saved predictions: {csv_path}")


def main():

    for language, path in DATASETS.items():
        evaluate_language(language, path)

    print(
        "\nMultiparagraph evaluation completed."
    )


if __name__ == "__main__":
    main()
