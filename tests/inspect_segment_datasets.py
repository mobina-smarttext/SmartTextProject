
import json
import re
from pathlib import Path
from collections import Counter


# ==========================================
# مسیر اصلی پروژه
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATHS = {
    "Persian": (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "test.jsonl"
    ),
    "English": (
        PROJECT_ROOT
        / "data"
        / "english"
        / "processed"
        / "test.jsonl"
    ),
}

MIN_SEGMENT_WORDS = 20


# ==========================================
# شمارش کلمات
# ==========================================

def count_words(text):
    return len(
        re.findall(
            r"[\w\u200c]+",
            text,
            flags=re.UNICODE
        )
    )


# ==========================================
# بررسی یک مجموعه‌داده
# ==========================================

def inspect_dataset(language, file_path):

    print("\n" + "=" * 60)
    print(f"Dataset language: {language}")
    print("=" * 60)

    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        return

    records = []

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1
        ):

            if not line.strip():
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError:
                print(
                    f"Invalid JSON at line {line_number}"
                )
                continue

            records.append(record)

    print(f"Total records: {len(records)}")

    if not records:
        print("Dataset is empty.")
        return

    # --------------------------------------
    # بررسی ساختار داده
    # --------------------------------------

    print("\nFirst record fields:")

    first_record = records[0]

    if not isinstance(first_record, dict):
        print(
            "Unexpected record format:",
            type(first_record).__name__
        )
        return

    for key in first_record.keys():
        print(f" - {key}")

    # --------------------------------------
    # شناسایی فیلد متن و برچسب
    # --------------------------------------

    possible_text_fields = [
        "text",
        "content",
        "response",
        "answer"
    ]

    possible_label_fields = [
        "label",
        "class",
        "target",
        "is_ai"
    ]

    text_field = next(
        (
            field
            for field in possible_text_fields
            if field in first_record
        ),
        None
    )

    label_field = next(
        (
            field
            for field in possible_label_fields
            if field in first_record
        ),
        None
    )

    print(f"\nDetected text field: {text_field}")
    print(f"Detected label field: {label_field}")

    if text_field is None:
        print("Cannot find text field.")
        return

    # --------------------------------------
    # بررسی طول متن‌ها
    # --------------------------------------

    word_counts = []

    short_count = 0
    valid_count = 0
    empty_count = 0

    labels = Counter()

    for record in records:

        if not isinstance(record, dict):
            continue

        text = record.get(text_field, "")

        if not isinstance(text, str):
            continue

        if not text.strip():
            empty_count += 1
            continue

        words = count_words(text)
        word_counts.append(words)

        if words < MIN_SEGMENT_WORDS:
            short_count += 1
        else:
            valid_count += 1

        if label_field is not None:
            label = record.get(label_field)
            labels[str(label)] += 1

    print("\nText statistics:")

    print(f"Non-empty texts: {len(word_counts)}")
    print(f"Empty texts: {empty_count}")
    print(f"Texts under 20 words: {short_count}")
    print(f"Texts with 20+ words: {valid_count}")

    if word_counts:

        print(
            f"Minimum word count: {min(word_counts)}"
        )

        print(
            f"Maximum word count: {max(word_counts)}"
        )

        print(
            "Average word count: "
            f"{sum(word_counts) / len(word_counts):.2f}"
        )

    # --------------------------------------
    # بررسی برچسب‌ها
    # --------------------------------------

    print("\nLabel distribution:")

    if labels:

        for label, count in sorted(labels.items()):
            print(
                f"Label {label}: {count}"
            )

    else:
        print("No label information found.")

    # --------------------------------------
    # بررسی جداکننده پاراگراف‌ها
    # --------------------------------------

    multi_paragraph_count = 0

    for record in records:

        if not isinstance(record, dict):
            continue

        text = record.get(text_field, "")

        if isinstance(text, str):
            paragraphs = re.split(
                r"\n\s*\n+",
                text.strip()
            )

            paragraphs = [
                p for p in paragraphs
                if p.strip()
            ]

            if len(paragraphs) > 1:
                multi_paragraph_count += 1

    print("\nParagraph structure:")

    print(
        "Records with multiple paragraphs: "
        f"{multi_paragraph_count}"
    )

    print(
        "Records with one paragraph: "
        f"{len(records) - multi_paragraph_count}"
    )

    print("\nDataset inspection completed.")


# ==========================================
# اجرای بررسی
# ==========================================

if __name__ == "__main__":

    for language, path in DATASET_PATHS.items():

        inspect_dataset(
            language,
            path
        )
