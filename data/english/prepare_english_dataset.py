
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "english" / "processed"

DATASET_URL = (
    "https://huggingface.co/datasets/"
    "Hello-SimpleAI/HC3/resolve/main/all.jsonl"
)

SEED = 42
MAX_QUESTION_GROUPS = 3000
MIN_WORDS = 30


def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()


def text_hash(text):
    normalized = normalize_text(text).casefold()
    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def first_valid_answer(answers):
    if not isinstance(answers, list):
        return None

    for answer in answers:
        if not isinstance(answer, str):
            continue

        cleaned = normalize_text(answer)

        if len(cleaned.split()) >= MIN_WORDS:
            return cleaned

    return None


def collect_groups(dataset):
    groups = []
    seen_questions = set()
    seen_texts = set()

    for row in dataset:
        question = row.get("question", "")

        if not isinstance(question, str):
            continue

        question = normalize_text(question)
        question_key = text_hash(question)

        if not question or question_key in seen_questions:
            continue

        human_text = first_valid_answer(
            row.get("human_answers")
        )
        ai_text = first_valid_answer(
            row.get("chatgpt_answers")
        )

        if not human_text or not ai_text:
            continue

        human_hash = text_hash(human_text)
        ai_hash = text_hash(ai_text)

        if (
            human_hash == ai_hash
            or human_hash in seen_texts
            or ai_hash in seen_texts
        ):
            continue

        seen_questions.add(question_key)
        seen_texts.update([human_hash, ai_hash])

        groups.append({
            "question_id": question_key,
            "source": row.get("source", "unknown"),
            "human_text": human_text,
            "ai_text": ai_text,
        })

    return groups


def write_split(groups, filename):
    output_path = OUTPUT_DIR / filename
    records = []

    for group in groups:
        records.append({
            "text": group["human_text"],
            "label": 0,
            "question_id": group["question_id"],
            "source": group["source"],
        })

        records.append({
            "text": group["ai_text"],
            "label": 1,
            "question_id": group["question_id"],
            "source": group["source"],
        })

    random.Random(SEED).shuffle(records)

    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    labels = Counter(record["label"] for record in records)

    print(f"\n{filename}")
    print(f"Total samples: {len(records)}")
    print(f"Human samples: {labels[0]}")
    print(f"AI samples: {labels[1]}")

    return records


def validate_splits(splits):
    question_sets = {}
    text_sets = {}

    for name, records in splits.items():
        question_sets[name] = {
            record["question_id"] for record in records
        }
        text_sets[name] = {
            text_hash(record["text"]) for record in records
        }

    names = list(splits)

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            first = names[i]
            second = names[j]

            if question_sets[first] & question_sets[second]:
                raise ValueError(
                    f"Question leakage: {first} and {second}"
                )

            if text_sets[first] & text_sets[second]:
                raise ValueError(
                    f"Duplicate text leakage: {first} and {second}"
                )

    print("\nValidation passed:")
    print("No exact question overlap between splits.")
    print("No normalized exact text overlap between splits.")


def main():
    print("Downloading HC3 English dataset...")

    dataset = load_dataset(
        "json",
        data_files={"train": DATASET_URL},
        split="train",
    )

    print(f"Original rows: {len(dataset)}")

    groups = collect_groups(dataset)

    print(f"Valid question groups: {len(groups)}")

    if len(groups) < 100:
        raise ValueError(
            "Not enough valid question groups for training."
        )

    rng = random.Random(SEED)
    rng.shuffle(groups)

    groups = groups[:MAX_QUESTION_GROUPS]

    total = len(groups)
    train_end = int(total * 0.70)
    val_end = int(total * 0.85)

    train_groups = groups[:train_end]
    val_groups = groups[train_end:val_end]
    test_groups = groups[val_end:]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": write_split(train_groups, "train.jsonl"),
        "val": write_split(val_groups, "val.jsonl"),
        "test": write_split(test_groups, "test.jsonl"),
    }

    validate_splits(splits)

    print("\nEnglish dataset preparation completed.")
    print(f"Files saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
