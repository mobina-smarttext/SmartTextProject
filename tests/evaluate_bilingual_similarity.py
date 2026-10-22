
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "modules" / "plagiarism")
)

from combined_similarity import compare_documents


TEST_CASES = [
    {
        "name": "English - Same topic",
        "text": (
            "Artificial intelligence has made significant "
            "progress in recent years. This technology is "
            "widely used in education and healthcare."
        ),
        "relevant_ids": {1, 2, 4, 5},
    },
    {
        "name": "Persian - Same topic",
        "text": (
            "هوش مصنوعی در سال‌های اخیر پیشرفت زیادی داشته است. "
            "این فناوری در آموزش و پزشکی کاربرد دارد."
        ),
        "relevant_ids": {1, 2, 4, 5},
    },
    {
        "name": "English - Air pollution",
        "text": (
            "Air pollution is a major problem in large cities. "
            "It can negatively affect human health."
        ),
        "relevant_ids": {3, 6},
    },
    {
        "name": "Persian - Air pollution",
        "text": (
            "آلودگی هوا یکی از مشکلات مهم شهرهای بزرگ است "
            "و می‌تواند بر سلامت انسان تأثیر بگذارد."
        ),
        "relevant_ids": {3, 6},
    },
]


def calculate_scores(result):
    lexical = result["lexical_similarity_percent"]
    semantic = result["semantic_similarity_percent"]

    current_score = 0.5 * lexical + 0.5 * semantic

    semantic_only = semantic

    semantic_weighted = (
        0.2 * lexical + 0.8 * semantic
    )

    return {
        "current": current_score,
        "semantic_only": semantic_only,
        "semantic_weighted": semantic_weighted,
    }


def evaluate_method(records, method_name):
    correct_pairs = 0
    total_pairs = 0

    relevant_scores = []
    unrelated_scores = []

    for record in records:
        for result in record["results"]:

            scores = calculate_scores(result)
            score = scores[method_name]

            is_relevant = (
                result["id"] in record["relevant_ids"]
            )

            if is_relevant:
                relevant_scores.append(score)
            else:
                unrelated_scores.append(score)

            total_pairs += 1

            # آستانه 50 درصد صرفاً آزمایشی است.
            predicted_relevant = score >= 50

            if predicted_relevant == is_relevant:
                correct_pairs += 1

    accuracy = (
        100 * correct_pairs / total_pairs
        if total_pairs else 0
    )

    relevant_average = (
        sum(relevant_scores) / len(relevant_scores)
        if relevant_scores else 0
    )

    unrelated_average = (
        sum(unrelated_scores) / len(unrelated_scores)
        if unrelated_scores else 0
    )

    return {
        "accuracy": accuracy,
        "relevant_average": relevant_average,
        "unrelated_average": unrelated_average,
    }


def main():
    print("=" * 65)
    print("BILINGUAL SIMILARITY EVALUATION")
    print("=" * 65)

    records = []

    for test_case in TEST_CASES:
        print(f"\nTEST: {test_case['name']}")
        print("-" * 65)

        results = compare_documents(test_case["text"])

        records.append({
            "name": test_case["name"],
            "results": results,
            "relevant_ids": test_case["relevant_ids"],
        })

        for result in results:
            scores = calculate_scores(result)

            print(
                f"{result['title']}: "
                f"Lexical={result['lexical_similarity_percent']:.2f}% | "
                f"Semantic={result['semantic_similarity_percent']:.2f}% | "
                f"Current={scores['current']:.2f}% | "
                f"SemanticOnly={scores['semantic_only']:.2f}% | "
                f"Weighted={scores['semantic_weighted']:.2f}%"
            )

    print("\n" + "=" * 65)
    print("METHOD COMPARISON")
    print("=" * 65)

    methods = {
        "current": "50% Lexical + 50% Semantic",
        "semantic_only": "100% Semantic",
        "semantic_weighted": "20% Lexical + 80% Semantic",
    }

    for method_key, method_title in methods.items():
        evaluation = evaluate_method(
            records,
            method_key
        )

        print(f"\nMethod: {method_title}")
        print(
            f"Accuracy at 50% threshold: "
            f"{evaluation['accuracy']:.2f}%"
        )
        print(
            f"Average relevant score: "
            f"{evaluation['relevant_average']:.2f}%"
        )
        print(
            f"Average unrelated score: "
            f"{evaluation['unrelated_average']:.2f}%"
        )


if __name__ == "__main__":
    main()
