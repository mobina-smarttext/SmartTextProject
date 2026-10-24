
from pathlib import Path
from datetime import datetime
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0, str(PROJECT_ROOT / "modules" / "plagiarism")
)

from combined_similarity import lexical_similarity
from model_loader import get_semantic_model


DATASET_PATH = (
    Path.home()
    / "Downloads"
    / "validation-00000-of-00001.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "similarity_english_external_evaluation_report.txt"
)

THRESHOLD = 0.40


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_parquet(DATASET_PATH)

    required = {"sentence1", "sentence2", "label"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"Unexpected dataset columns: {list(df.columns)}"
        )

    df = df.dropna(
        subset=["sentence1", "sentence2", "label"]
    )

    df = df[
        df["sentence1"].astype(str).str.strip().ne("")
        & df["sentence2"].astype(str).str.strip().ne("")
    ]

    df = df[df["label"].isin([0, 1])]

    if df.empty or df["label"].nunique() != 2:
        raise ValueError(
            "Dataset must contain both classes."
        )

    pairs = list(
        zip(
            df["sentence1"].astype(str),
            df["sentence2"].astype(str),
            df["label"].astype(int),
        )
    )

    model = get_semantic_model()

    unique_texts = list(
        dict.fromkeys(
            text
            for text1, text2, _ in pairs
            for text in (text1, text2)
        )
    )

    embeddings = model.encode(
        unique_texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embedding_map = dict(
        zip(unique_texts, embeddings)
    )

    scores = []
    labels = []

    for text1, text2, label in pairs:
        lexical = lexical_similarity(text1, text2)

        semantic = float(
            np.dot(
                embedding_map[text1],
                embedding_map[text2],
            )
        )

        semantic = max(0.0, min(1.0, semantic))

        combined = 0.5 * lexical + 0.5 * semantic

        scores.append(combined)
        labels.append(label)

    # Analyze similarity score distributions
    distribution_lines = [
        "",
        "SIMILARITY SCORE DISTRIBUTIONS:",
    ]

    for label, name in [
        (0, "Non-paraphrase"),
        (1, "Paraphrase"),
    ]:
        group = sorted(
            score
            for score, actual in zip(scores, labels)
            if actual == label
        )

        distribution_lines.extend([
            "",
            f"{name}:",
            f"Count: {len(group)}",
            f"Minimum: {group[0]:.4f}",
            f"25th percentile: {np.percentile(group, 25):.4f}",
            f"Median: {np.median(group):.4f}",
            f"75th percentile: {np.percentile(group, 75):.4f}",
            f"Maximum: {group[-1]:.4f}",
            f"Mean: {np.mean(group):.4f}",
        ])

    # Evaluate the original threshold
    predictions = [
        int(score >= THRESHOLD)
        for score in scores
    ]

    accuracy = accuracy_score(labels, predictions)

    precision = precision_score(
        labels, predictions, zero_division=0
    )

    recall = recall_score(
        labels, predictions, zero_division=0
    )

    f1 = f1_score(
        labels, predictions, zero_division=0
    )

    auc = roc_auc_score(labels, scores)

    matrix = confusion_matrix(
        labels, predictions, labels=[0, 1]
    )

    # Compare exploratory thresholds
    comparison_lines = [
        "",
        "THRESHOLD COMPARISON:",
        "Threshold | Accuracy | Precision | Recall | F1 | FP | FN",
    ]

    for threshold in [0.40, 0.50, 0.60, 0.70]:
        preds = [
            int(score >= threshold)
            for score in scores
        ]

        tn, fp, fn, tp = confusion_matrix(
            labels, preds, labels=[0, 1]
        ).ravel()

        comparison_lines.append(
            f"{threshold:.0%} | "
            f"{accuracy_score(labels, preds):.4f} | "
            f"{precision_score(labels, preds, zero_division=0):.4f} | "
            f"{recall_score(labels, preds, zero_division=0):.4f} | "
            f"{f1_score(labels, preds, zero_division=0):.4f} | "
            f"{fp} | {fn}"
        )

    lines = [
        "External English Similarity Evaluation",
        "=" * 60,
        f"Date: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "Dataset: GLUE MRPC validation",
        f"Samples: {len(pairs)}",
        f"Positive: {sum(labels)}",
        f"Negative: {len(labels) - sum(labels)}",
        f"Threshold (preselected): {THRESHOLD:.0%}",
        "",
        f"Accuracy: {accuracy:.4f}",
        f"Precision: {precision:.4f}",
        f"Recall: {recall:.4f}",
        f"F1: {f1:.4f}",
        f"ROC AUC: {auc:.4f}",
        "",
        "Confusion matrix (rows=true, columns=predicted):",
        str(matrix),
    ]

    lines.extend(distribution_lines)
    lines.extend(comparison_lines)

    lines.extend([
        "",
        "LIMITATIONS:",
        "- Public labeled dataset; training overlap not verified.",
        "- MRPC measures paraphrase equivalence, not plagiarism.",
        "- Threshold comparison is exploratory on the same data.",
        "- No threshold is independently calibrated.",
        "- Dataset domain differs from student documents.",
        "- Similarity does not establish plagiarism.",
    ])

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    report = "\n".join(lines)

    OUTPUT_PATH.write_text(
        report,
        encoding="utf-8-sig"
    )

    print(report)
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
