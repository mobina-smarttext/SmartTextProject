
import csv
import gzip
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0, str(PROJECT_ROOT / "modules" / "plagiarism")
)

from combined_similarity import lexical_similarity
from model_loader import get_semantic_model

DATASET_PATH = (
    Path.home() / "Downloads" / "ExaPPC.txt.gz"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "reports"
    / "similarity_external_evaluation_report.txt"
)

SAMPLES_PER_CLASS = 100
RANDOM_SEED = 42
THRESHOLD = 0.40


def load_samples():
    rng = random.Random(RANDOM_SEED)
    samples = {
        "paraphrase": [],
        "non-paraphrase": [],
    }
    counts = {label: 0 for label in samples}

    with gzip.open(
        DATASET_PATH, "rt", encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file, delimiter="\t")

        for row in reader:
            label = row.get("label")
            if label not in samples:
                continue

            text1 = (row.get("sentence1") or "").strip()
            text2 = (row.get("sentence2") or "").strip()

            if not text1 or not text2:
                continue

            counts[label] += 1
            seen = counts[label]
            bucket = samples[label]

            if len(bucket) < SAMPLES_PER_CLASS:
                bucket.append((text1, text2))
            else:
                index = rng.randrange(seen)
                if index < SAMPLES_PER_CLASS:
                    bucket[index] = (text1, text2)

    if any(
        len(bucket) < SAMPLES_PER_CLASS
        for bucket in samples.values()
    ):
        raise ValueError("Not enough valid samples.")

    selected = [
        (a, b, 1)
        for a, b in samples["paraphrase"]
    ] + [
        (a, b, 0)
        for a, b in samples["non-paraphrase"]
    ]

    rng.shuffle(selected)
    return selected


def main():
    pairs = load_samples()

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

    lines = [
        "External Persian Similarity Evaluation",
        "=" * 60,
        f"Date: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "Dataset: ExaPPC",
        f"Random seed: {RANDOM_SEED}",
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
        "",
        "LIMITATIONS:",
        "- Public labeled dataset; not a verified independent test of training-data overlap.",
        "- Only 200 randomly sampled pairs.",
        "- Paraphrase labels are not plagiarism labels.",
        "- Dataset domain may differ from student documents.",
        "- Threshold is exploratory and not calibrated.",
        "- Similarity does not establish plagiarism.",
    ]

    OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )
    report = "\n".join(lines)
    OUTPUT_PATH.write_text(
        report, encoding="utf-8-sig"
    )

    print(report)
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
