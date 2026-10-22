
import json
import re
import hashlib
import unicodedata
from pathlib import Path
from collections import defaultdict


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "Persian": {
        "train": PROJECT_ROOT / "data" / "processed" / "train.jsonl",
        "val": PROJECT_ROOT / "data" / "processed" / "val.jsonl",
        "test": PROJECT_ROOT / "data" / "processed" / "test.jsonl",
    },
    "English": {
        "train": PROJECT_ROOT / "data" / "english" / "processed" / "train.jsonl",
        "val": PROJECT_ROOT / "data" / "english" / "processed" / "val.jsonl",
        "test": PROJECT_ROOT / "data" / "english" / "processed" / "test.jsonl",
    },
}

PAIRS = [
    ("train", "val"),
    ("train", "test"),
    ("val", "test"),
]


def load_jsonl(path):
    records = []

    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)

            if not isinstance(record, dict):
                raise ValueError(
                    f"Invalid record at {path.name}:{line_number}"
                )

            records.append(record)

    return records


def normalize_text(text):
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)

    # یکسان‌سازی برخی حروف فارسی و عربی
    text = text.replace("ي", "ی").replace("ك", "ک")

    # حذف فاصله‌های اضافی
    text = re.sub(r"\s+", " ", text)

    return text.strip().casefold()


def text_fingerprint(text):
    normalized = normalize_text(text)

    if not normalized:
        return None

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def get_values(records, field):
    values = set()

    for record in records:
        value = record.get(field)

        if value is None or value == "":
            continue

        # تبدیل به رشته برای مقایسه شناسه‌ها
        values.add(str(value))

    return values


def get_text_hashes(records):
    hashes = set()

    for record in records:
        fingerprint = text_fingerprint(
            record.get("text", "")
        )

        if fingerprint:
            hashes.add(fingerprint)

    return hashes


def inspect_language(language, paths):

    print("\n" + "=" * 65)
    print(f"DATA LEAKAGE CHECK: {language}")
    print("=" * 65)

    datasets = {
        split: load_jsonl(path)
        for split, path in paths.items()
    }

    for split, records in datasets.items():
        print(f"{split}: {len(records)} records")

    fields = (
        ["id", "human_id"]
        if language == "Persian"
        else ["question_id"]
    )

    print("\n--- Exact normalized text overlap ---")

    hashes = {
        split: get_text_hashes(records)
        for split, records in datasets.items()
    }

    for first, second in PAIRS:
        overlap = hashes[first] & hashes[second]

        print(
            f"{first} vs {second}: "
            f"{len(overlap)} shared text fingerprints"
        )

    print("\n--- Identifier overlap ---")

    for field in fields:

        print(f"\nField: {field}")

        values = {
            split: get_values(records, field)
            for split, records in datasets.items()
        }

        for first, second in PAIRS:

            overlap = values[first] & values[second]

            print(
                f"{first} vs {second}: "
                f"{len(overlap)} shared identifiers"
            )

    print("\n--- Duplicate texts within each split ---")

    for split, records in datasets.items():

        fingerprints = [
            text_fingerprint(record.get("text", ""))
            for record in records
        ]

        fingerprints = [
            value for value in fingerprints
            if value is not None
        ]

        duplicate_count = (
            len(fingerprints) - len(set(fingerprints))
        )

        print(
            f"{split}: {duplicate_count} duplicate records"
        )

    print("\nCheck completed.")


if __name__ == "__main__":

    for language, paths in DATASETS.items():
        inspect_language(language, paths)
