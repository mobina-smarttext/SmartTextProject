
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "modules" / "plagiarism")
)

from combined_similarity import compare_documents


TEST_CASES = [
    {
        "name": "Identical text",
        "text": (
            "هوش مصنوعی در سال‌های اخیر پیشرفت زیادی داشته است. "
            "این فناوری در آموزش و پزشکی کاربرد دارد."
        ),
        "expected_documents": ["متن اول"]
    },
    {
        "name": "Paraphrased AI text",
        "text": (
            "در سال‌های اخیر فناوری هوش مصنوعی رشد چشمگیری داشته "
            "و در حوزه‌های پزشکی و آموزش مورد استفاده قرار می‌گیرد."
        ),
        "expected_documents": ["متن اول", "متن دوم"]
    },
    {
        "name": "Paraphrased air pollution text",
        "text": (
            "آلودگی هوا در شهرهای بزرگ مشکلات زیادی برای "
            "سلامت انسان ایجاد می‌کند."
        ),
        "expected_documents": ["متن سوم"]
    },
    {
        "name": "Different topic",
        "text": (
            "تیم فوتبال در مسابقه دیروز موفق شد "
            "با سه گل حریف خود را شکست دهد."
        ),
        "expected_documents": []
    }
]


def main():
    lines = [
        "Similarity Evaluation Report",
        "=" * 60,
        f"Date: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "Method: TF-IDF + Semantic Similarity",
        "Combined score: 50% lexical + 50% semantic",
        "",
    ]

    passed = 0
    evaluated = 0

    for case in TEST_CASES:
        lines.append("-" * 60)
        lines.append(f"Test: {case['name']}")
        lines.append(f"Input: {case['text']}")
        lines.append("")

        results = compare_documents(case["text"])

        lines.append("Document scores:")

        for result in results:
            lines.append(
                f"{result['title']}: "
                f"Lexical={result['lexical_similarity_percent']}%, "
                f"Semantic={result['semantic_similarity_percent']}%, "
                f"Combined={result['combined_similarity_percent']}%"
            )

        if not results:
            lines.append("Result: NO DOCUMENTS")
            if case["expected_documents"]:
                evaluated += 1
            lines.append("")
            continue

        top_document = results[0]["title"]
        lines.append(f"Top document: {top_document}")

        if case["expected_documents"]:
            evaluated += 1

            if top_document in case["expected_documents"]:
                passed += 1
                lines.append("Result: PASS")
            else:
                lines.append("Result: FAIL")
        else:
            lines.append("Result: OBSERVATION ONLY")

        lines.append("")

    lines.extend([
        "=" * 60,
        f"Passed: {passed}/{evaluated} labeled ranking tests",
        "Different-topic test: observation only",
        "",
        "Limitations:",
        "- Only a small set of manually designed tests was used.",
        "- This is not a statistical accuracy estimate.",
        "- No calibrated similarity threshold has been established.",
        "- Similarity does not prove plagiarism.",
    ])

    report_text = "\n".join(lines)

    reports_folder = PROJECT_ROOT / "reports"
    reports_folder.mkdir(parents=True, exist_ok=True)

    output_file = (
        reports_folder / "similarity_evaluation_report.txt"
    )

    output_file.write_text(
        report_text,
        encoding="utf-8-sig"
    )

    print(report_text)
    print("\nReport saved successfully!")
    print(f"File: {output_file}")


if __name__ == "__main__":
    main()
