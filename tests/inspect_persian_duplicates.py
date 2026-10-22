
import json
import re
import unicodedata
from pathlib import Path
from collections import defaultdict


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"


def normalize_text(text):

    text = unicodedata.normalize("NFKC", text)

    text = text.replace("ي", "ی")
    text = text.replace("ك", "ک")

    text = re.sub(r"\s+", " ", text)

    return text.strip().casefold()


def load_dataset(filename):

    records = []

    path = DATA_DIR / filename

    with open(path, "r", encoding="utf-8") as file:

        for line in file:

            if line.strip():
                records.append(json.loads(line))

    return records


def group_by_text(records):

    groups = defaultdict(list)

    for record in records:

        text = record.get("text", "")

        if not isinstance(text, str):
            continue

        normalized = normalize_text(text)

        if normalized:
            groups[normalized].append(record)

    return groups


def print_record_info(record):

    fields = [
        "id",
        "human_id",
        "label",
        "source",
        "model",
        "category",
        "subcategory",
        "num_words",
    ]

    for field in fields:
        print(f"  {field}: {record.get(field)}")

    print("-" * 45)


def main():

    train = load_dataset("train.jsonl")
    val = load_dataset("val.jsonl")

    train_groups = group_by_text(train)
    val_groups = group_by_text(val)

    print("\n" + "=" * 60)
    print("TRAIN / VALIDATION DUPLICATES")
    print("=" * 60)

    shared_texts = (
        set(train_groups.keys())
        & set(val_groups.keys())
    )

    print(f"Shared unique texts: {len(shared_texts)}")

    for index, text in enumerate(
        sorted(shared_texts),
        start=1
    ):

        print(f"\nShared text group #{index}")

        print("\nTRAIN RECORDS:")

        for record in train_groups[text]:
            print_record_info(record)

        print("\nVALIDATION RECORDS:")

        for record in val_groups[text]:
            print_record_info(record)

        labels = {
            record.get("label")
            for record in (
                train_groups[text] + val_groups[text]
            )
        }

        print(f"Labels in group: {labels}")

    print("\n" + "=" * 60)
    print("DUPLICATES INSIDE TRAIN")
    print("=" * 60)

    duplicate_groups = [
        group
        for group in train_groups.values()
        if len(group) > 1
    ]

    print(
        f"Duplicate groups: {len(duplicate_groups)}"
    )

    for index, group in enumerate(
        duplicate_groups,
        start=1
    ):

        print(f"\nDuplicate group #{index}")
        print(f"Number of records: {len(group)}")

        for record in group:
            print_record_info(record)

        labels = {
            record.get("label")
            for record in group
        }

        print(f"Labels in group: {labels}")


if __name__ == "__main__":
    main()
