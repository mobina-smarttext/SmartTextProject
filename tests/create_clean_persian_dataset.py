
import json
import hashlib
import re
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict


# ==========================================
# مسیرهای پروژه
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "cleaned_experimental"
    / "persian"
)

SPLITS = ("train", "val", "test")

# گروهی که باید از اعتبارسنجی حذف شود
EXCLUDED_VALIDATION_GROUP = "0296"

# گروه‌هایی که فعلاً حفظ می‌شوند
KNOWN_DUPLICATE_PAIRS = [
    ("0947", "0636"),
    ("0091", "0306"),
    ("0215", "0059"),
]

EXPECTED_ORIGINAL_COUNTS = {
    "train": 2800,
    "val": 400,
    "test": 800,
}


# ==========================================
# توابع کمکی
# ==========================================

def normalize_text(text):
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("ي", "ی")
    text = text.replace("ك", "ک")
    text = re.sub(r"\s+", " ", text)

    return text.strip().casefold()


def text_hash(text):
    normalized = normalize_text(text)

    if not normalized:
        return None

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def load_jsonl(path):
    records = []

    if not path.exists():
        raise FileNotFoundError(
            f"Source file not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1
        ):

            if not line.strip():
                continue

            record = json.loads(line)

            if not isinstance(record, dict):
                raise ValueError(
                    f"Invalid record at "
                    f"{path.name}:{line_number}"
                )

            records.append(record)

    return records


def write_jsonl(path, records):
    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )


def get_human_id(record):
    value = record.get("human_id")

    if value is None:
        raise ValueError(
            "A record has no human_id."
        )

    return str(value)


def validate_group_structure(datasets):

    groups = defaultdict(list)

    for split, records in datasets.items():
        for record in records:
            groups[get_human_id(record)].append(
                (split, record)
            )

    for human_id, items in groups.items():

        labels = Counter(
            record.get("label")
            for _, record in items
        )

        splits = {
            split
            for split, _ in items
        }

        if len(splits) != 1:
            raise ValueError(
                f"Group {human_id} crosses splits."
            )

        if (
            len(items) != 4
            or labels.get(0, 0) != 1
            or labels.get(1, 0) != 3
        ):
            raise ValueError(
                f"Invalid group structure: {human_id}"
            )

    return len(groups)


def get_text_hashes(records):

    return {
        fingerprint
        for record in records
        if (
            fingerprint := text_hash(
                record.get("text", "")
            )
        ) is not None
    }


def validate_no_cross_split_duplicates(datasets):

    hashes = {
        split: get_text_hashes(records)
        for split, records in datasets.items()
    }

    pairs = [
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ]

    for first, second in pairs:

        overlap = (
            hashes[first] & hashes[second]
        )

        if overlap:
            raise ValueError(
                f"Found {len(overlap)} duplicated "
                f"texts between {first} and {second}."
            )


def count_labels(records):

    return dict(
        sorted(
            Counter(
                str(record.get("label"))
                for record in records
            ).items()
        )
    )


# ==========================================
# ساخت نسخه آزمایشی
# ==========================================

def main():

    print("\n" + "=" * 65)
    print("CREATE CLEAN EXPERIMENTAL PERSIAN DATASET")
    print("=" * 65)

    # خواندن فایل‌های اصلی
    original = {
        split: load_jsonl(
            SOURCE_DIR / f"{split}.jsonl"
        )
        for split in SPLITS
    }

    print("\nOriginal dataset counts:")

    for split in SPLITS:

        count = len(original[split])

        print(f"{split}: {count}")

        expected = EXPECTED_ORIGINAL_COUNTS[split]

        if count != expected:
            raise ValueError(
                f"Unexpected {split} size: "
                f"{count}, expected {expected}"
            )

    # بررسی سلامت گروه‌بندی اصلی
    original_group_count = (
        validate_group_structure(original)
    )

    print(
        "\nOriginal human_id groups:",
        original_group_count
    )

    # اطمینان از وجود گروه هدف در اعتبارسنجی
    target_records = [
        record
        for record in original["val"]
        if get_human_id(record)
        == EXCLUDED_VALIDATION_GROUP
    ]

    target_labels = Counter(
        record.get("label")
        for record in target_records
    )

    if (
        len(target_records) != 4
        or target_labels.get(0, 0) != 1
        or target_labels.get(1, 0) != 3
    ):
        raise ValueError(
            "Validation group 0296 does not "
            "have the expected structure."
        )

    # ساخت نسخه جدید در حافظه
    cleaned = {
        "train": list(original["train"]),

        "val": [
            record
            for record in original["val"]
            if get_human_id(record)
            != EXCLUDED_VALIDATION_GROUP
        ],

        "test": list(original["test"]),
    }

    # بررسی تعداد رکوردها
    expected_cleaned_counts = {
        "train": 2800,
        "val": 396,
        "test": 800,
    }

    for split in SPLITS:

        actual = len(cleaned[split])
        expected = expected_cleaned_counts[split]

        if actual != expected:
            raise ValueError(
                f"Unexpected cleaned {split} count: "
                f"{actual}, expected {expected}"
            )

    # بررسی ساختار گروه‌های جدید
    cleaned_group_count = (
        validate_group_structure(cleaned)
    )

    if cleaned_group_count != 999:
        raise ValueError(
            "Expected 999 cleaned groups, "
            f"got {cleaned_group_count}."
        )

    # بررسی نبود متن یکسان بین مجموعه‌ها
    validate_no_cross_split_duplicates(cleaned)

    # اطمینان از حفظ کامل train و test
    if cleaned["train"] != original["train"]:
        raise ValueError(
            "Train data changed unexpectedly."
        )

    if cleaned["test"] != original["test"]:
        raise ValueError(
            "Test data changed unexpectedly."
        )

    # اطمینان از حذف کامل گروه هدف
    for split in SPLITS:

        remaining = [
            record
            for record in cleaned[split]
            if get_human_id(record)
            == EXCLUDED_VALIDATION_GROUP
        ]

        if remaining:
            raise ValueError(
                "Excluded group still exists "
                f"in {split}."
            )

    # فقط پس از موفقیت همه بررسی‌ها
    # فایل‌های خروجی ساخته می‌شوند
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for split in SPLITS:

        output_path = (
            OUTPUT_DIR / f"{split}.jsonl"
        )

        write_jsonl(
            output_path,
            cleaned[split]
        )

    # ساخت گزارش
    report = {
        "method": "conservative_group_cleanup",
        "source_directory": str(SOURCE_DIR),
        "output_directory": str(OUTPUT_DIR),
        "excluded_validation_human_id": (
            EXCLUDED_VALIDATION_GROUP
        ),
        "excluded_records": len(target_records),
        "original_group_count": original_group_count,
        "cleaned_group_count": cleaned_group_count,
        "known_duplicate_pairs_retained": [
            list(pair)
            for pair in KNOWN_DUPLICATE_PAIRS
        ],
        "original_counts": {
            split: len(original[split])
            for split in SPLITS
        },
        "cleaned_counts": {
            split: len(cleaned[split])
            for split in SPLITS
        },
        "original_label_counts": {
            split: count_labels(original[split])
            for split in SPLITS
        },
        "cleaned_label_counts": {
            split: count_labels(cleaned[split])
            for split in SPLITS
        },
        "notes": [
            "Original datasets were not modified.",
            "Original models were not modified.",
            "Train and test records were preserved.",
            "Group 0296 was excluded from validation.",
            "Three duplicate human-text pairs "
            "inside train were intentionally retained.",
            "Exact normalized text overlap across "
            "splits was checked.",
            "Semantic or source-level independence "
            "has not been established.",
            "Existing calibrated model was fitted "
            "using the original validation set.",
            "A new calibration is required before "
            "claiming cleaned-validation results."
        ],
    }

    report_path = (
        OUTPUT_DIR / "cleanup_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2
        )

    # نمایش نتیجه
    print("\nCleaned dataset counts:")

    for split in SPLITS:

        print(
            f"{split}: {len(cleaned[split])}"
        )

        print(
            "  labels:",
            count_labels(cleaned[split])
        )

    print(
        "\nCleaned human_id groups:",
        cleaned_group_count
    )

    print(
        "Removed validation group:",
        EXCLUDED_VALIDATION_GROUP
    )

    print(
        "Removed records:",
        len(target_records)
    )

    print(
        "\nCross-split exact-text overlap: 0"
    )

    print(
        "\nOriginal datasets: UNCHANGED"
    )

    print(
        "Original models: UNCHANGED"
    )

    print(
        "\nOutput directory:",
        OUTPUT_DIR
    )

    print(
        "\nCleanup completed successfully."
    )


if __name__ == "__main__":
    main()
