
import json
import random
import re
from pathlib import Path
from collections import Counter


# ==========================================
# تنظیمات
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "Persian": (
        PROJECT_ROOT
        / "data"
        / "cleaned_experimental"
        / "persian"
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "multiparagraph"
)

DOCUMENTS_PER_LANGUAGE = 50
SEGMENTS_PER_DOCUMENT = 4

RANDOM_SEED = 42

# الگوی برچسب‌ها:
# 0 = انسانی
# 1 = AI
LABEL_PATTERN = [0, 1, 0, 1]


# ==========================================
# توابع کمکی
# ==========================================

def load_jsonl(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    records = []

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
                    f"Invalid record at line {line_number}"
                )

            records.append(record)

    return records


def count_words(text):

    return len(
        re.findall(
            r"[\w\u200c]+",
            text,
            flags=re.UNICODE
        )
    )


def prepare_records(records, language):

    group_field = (
        "human_id"
        if language == "Persian"
        else "question_id"
    )

    prepared = []

    for index, record in enumerate(records):

        text = record.get("text")
        label = record.get("label")
        group_id = record.get(group_field)

        if not isinstance(text, str):
            continue

        text = text.strip()

        if not text:
            continue

        if type(label) is not int:
            continue

        if label not in (0, 1):
            continue

        if group_id is None:
            continue

        if count_words(text) < 20:
            continue

        # جلوگیری از وجود جداکننده پاراگراف
        # درون هر نمونه
        if re.search(r"\n\s*\n+", text):
            continue

        prepared.append({
            "text": text,
            "label": label,
            "group_id": str(group_id),
            "original_id": str(
                record.get("id", index)
            ),
            "source": record.get("source"),
        })

    return prepared


# ==========================================
# ساخت سندهای ترکیبی
# ==========================================

def build_documents(language, records):

    rng = random.Random(RANDOM_SEED)

    prepared = prepare_records(
        records,
        language
    )

    human_records = [
        record
        for record in prepared
        if record["label"] == 0
    ]

    ai_records = [
        record
        for record in prepared
        if record["label"] == 1
    ]

    rng.shuffle(human_records)
    rng.shuffle(ai_records)

    used_groups = set()
    used_records = set()

    documents = []

    for document_index in range(
        DOCUMENTS_PER_LANGUAGE
    ):

        selected_segments = []

        for required_label in LABEL_PATTERN:

            candidates = (
                human_records
                if required_label == 0
                else ai_records
            )

            chosen = None

            for candidate in candidates:

                record_key = (
                    candidate["original_id"],
                    candidate["group_id"],
                    candidate["label"]
                )

                if record_key in used_records:
                    continue

                if candidate["group_id"] in used_groups:
                    continue

                chosen = candidate
                break

            if chosen is None:
                raise ValueError(
                    f"Not enough independent "
                    f"{language} samples for "
                    f"{DOCUMENTS_PER_LANGUAGE} documents."
                )

            selected_segments.append(chosen)

            used_groups.add(
                chosen["group_id"]
            )

            used_records.add(
                (
                    chosen["original_id"],
                    chosen["group_id"],
                    chosen["label"]
                )
            )

        document_text = "\n\n".join(
            segment["text"]
            for segment in selected_segments
        )

        document = {
            "document_id": (
                f"{language.lower()}_mixed_"
                f"{document_index + 1:03d}"
            ),
            "language": language,
            "document_type": "synthetic_mixed",
            "text": document_text,
            "segments": [
                {
                    "segment_number": index + 1,
                    "label": segment["label"],
                    "group_id": segment["group_id"],
                    "original_id": segment["original_id"],
                    "text": segment["text"],
                    "word_count": count_words(
                        segment["text"]
                    ),
                }
                for index, segment in enumerate(
                    selected_segments
                )
            ],
        }

        documents.append(document)

    return documents


# ==========================================
# اعتبارسنجی خروجی
# ==========================================

def validate_documents(documents, language):

    if len(documents) != DOCUMENTS_PER_LANGUAGE:
        raise ValueError(
            "Unexpected number of documents."
        )

    all_group_ids = []
    all_labels = []

    for document in documents:

        segments = document["segments"]

        if len(segments) != SEGMENTS_PER_DOCUMENT:
            raise ValueError(
                "Unexpected segment count."
            )

        labels = [
            segment["label"]
            for segment in segments
        ]

        if labels != LABEL_PATTERN:
            raise ValueError(
                "Unexpected label pattern."
            )

        expected_text = "\n\n".join(
            segment["text"]
            for segment in segments
        )

        if document["text"] != expected_text:
            raise ValueError(
                "Document text mismatch."
            )

        for segment in segments:

            if segment["word_count"] < 20:
                raise ValueError(
                    "Segment shorter than 20 words."
                )

            all_group_ids.append(
                segment["group_id"]
            )

            all_labels.append(
                segment["label"]
            )

    if len(all_group_ids) != len(set(all_group_ids)):
        raise ValueError(
            "A source group was reused."
        )

    label_counts = Counter(all_labels)

    print(f"\nValidation for {language}:")
    print(f"Documents: {len(documents)}")
    print(f"Segments: {len(all_labels)}")
    print(f"Human segments: {label_counts[0]}")
    print(f"AI segments: {label_counts[1]}")
    print(
        "Unique source groups:",
        len(set(all_group_ids))
    )

    print("Validation passed.")


# ==========================================
# ذخیره خروجی
# ==========================================

def save_documents(language, documents):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR
        / f"{language.lower()}_multiparagraph.jsonl"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        for document in documents:

            file.write(
                json.dumps(
                    document,
                    ensure_ascii=False
                ) + "\n"
            )

    print(f"Saved: {output_path}")


# ==========================================
# اجرای اصلی
# ==========================================

def main():

    print("\n" + "=" * 65)
    print("CREATE MULTIPARAGRAPH EVALUATION DATASET")
    print("=" * 65)

    for language, path in DATASETS.items():

        print(f"\nProcessing {language}...")

        records = load_jsonl(path)

        print(
            "Original test records:",
            len(records)
        )

        documents = build_documents(
            language,
            records
        )

        validate_documents(
            documents,
            language
        )

        save_documents(
            language,
            documents
        )

    print(
        "\nMultiparagraph dataset creation completed."
    )


if __name__ == "__main__":
    main()
