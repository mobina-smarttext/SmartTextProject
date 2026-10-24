
from pathlib import Path
from datetime import datetime
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "modules" / "plagiarism")
)

from combined_similarity import compare_documents


TEST_CASES = [
    {
        "name": "Identical Persian AI text",
        "text": "هوش مصنوعی در سال‌های اخیر پیشرفت زیادی داشته است. این فناوری در آموزش و پزشکی کاربرد دارد.",
        "relevant_ids": [1, 2],
    },
    {
        "name": "Paraphrased Persian AI text",
        "text": "در سال‌های اخیر فناوری هوش مصنوعی رشد چشمگیری داشته و در حوزه‌های پزشکی و آموزش مورد استفاده قرار می‌گیرد.",
        "relevant_ids": [1, 2],
    },
    {
        "name": "Paraphrased air pollution",
        "text": "آلودگی هوا در شهرهای بزرگ مشکلات زیادی برای سلامت انسان ایجاد می‌کند.",
        "relevant_ids": [3],
    },
    {
        "name": "Unrelated Persian football text",
        "text": "تیم فوتبال در مسابقه دیروز موفق شد با سه گل حریف خود را شکست دهد.",
        "relevant_ids": [],
    },
    {
        "name": "Identical English AI text",
        "text": "Artificial intelligence helps computers identify patterns in data. It is widely used in natural language processing, education, and information analysis.",
        "relevant_ids": [4],
    },
    {
        "name": "English machine learning paraphrase",
        "text": "Machine learning systems study examples and discover patterns to predict outcomes for new data.",
        "relevant_ids": [5],
    },
    {
        "name": "English plagiarism paraphrase",
        "text": "Academic plagiarism checkers compare written documents to find overlapping passages, but human review is still necessary.",
        "relevant_ids": [6],
    },
    {
        "name": "Unrelated English sports text",
        "text": "The basketball team won the championship after a difficult final match.",
        "relevant_ids": [],
    },
]

THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]


def metrics(tp, fp, fn, tn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    accuracy = (
        (tp + tn) / (tp + fp + fn + tn)
        if tp + fp + fn + tn else 0.0
    )
    return precision, recall, f1, accuracy


def main():
    documents_path = (
        PROJECT_ROOT / "data" / "plagiarism" / "documents.json"
    )

    documents = json.loads(
        documents_path.read_text(encoding="utf-8-sig")
    )

    document_ids = {doc["id"] for doc in documents}

    lines = [
        "Extended Similarity Evaluation",
        "=" * 60,
        f"Date: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Queries: {len(TEST_CASES)}",
        f"Reference documents: {len(documents)}",
        "Labels: manually assigned, exploratory",
        "",
    ]

    pairs = []
    ranking_hits = 0
    ranking_total = 0

    for case in TEST_CASES:
        relevant = set(case["relevant_ids"])

        if not relevant.issubset(document_ids):
            raise ValueError(
                f"Unknown document ID in {case['name']}"
            )

        results = compare_documents(case["text"])

        returned_ids = {result["id"] for result in results}

        if returned_ids != document_ids:
            raise ValueError(
                f"Incomplete results for {case['name']}"
            )

        top_id = results[0]["id"] if results else None

        if relevant:
            ranking_total += 1
            if top_id in relevant:
                ranking_hits += 1

        lines.append(f"Test: {case['name']}")
        lines.append(f"Relevant IDs: {sorted(relevant)}")
        lines.append(f"Top document ID: {top_id}")

        for result in results:
            score = result["combined_similarity_percent"] / 100

            pairs.append({
                "query": case["name"],
                "document_id": result["id"],
                "score": score,
                "label": int(result["id"] in relevant),
            })

        lines.append("")

    lines.append(
        f"Top-1 relevant ranking: "
        f"{ranking_hits}/{ranking_total}"
    )
    lines.append("")

    for threshold in THRESHOLDS:
        tp = fp = fn = tn = 0

        for pair in pairs:
            predicted = int(pair["score"] >= threshold)
            actual = pair["label"]

            if predicted == 1 and actual == 1:
                tp += 1
            elif predicted == 1 and actual == 0:
                fp += 1
            elif predicted == 0 and actual == 1:
                fn += 1
            else:
                tn += 1

        precision, recall, f1, accuracy = metrics(
            tp, fp, fn, tn
        )

        lines.extend([
            f"Threshold: {threshold:.0%}",
            f"TP={tp}, FP={fp}, FN={fn}, TN={tn}",
            f"Precision: {precision:.4f}",
            f"Recall: {recall:.4f}",
            f"F1: {f1:.4f}",
            f"Accuracy: {accuracy:.4f}",
            "",
        ])

    lines.extend([
        "IMPORTANT LIMITATIONS:",
        "- Small manually constructed evaluation set.",
        "- Queries and reference documents are not independent.",
        "- Repeated document pairs are not independent observations.",
        "- Some relevance labels are subjective.",
        "- Thresholds are exploratory, not calibrated.",
        "- Do not select a production threshold from these results.",
        "- Results are not external validation.",
        "- Similarity does not prove plagiarism.",
    ])

    report = "\n".join(lines)

    output = (
        PROJECT_ROOT
        / "reports"
        / "similarity_extended_evaluation_report.txt"
    )

    output.write_text(report, encoding="utf-8-sig")

    print(report)
    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
