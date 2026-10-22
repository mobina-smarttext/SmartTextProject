
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "raw"
SPLITS = ["train.jsonl", "val.jsonl", "test.jsonl"]


def load_records(filename):
    path = DATA_DIR / filename

    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


records = {
    name: load_records(name)
    for name in SPLITS
}

# بررسی شناسه رکوردها
print("=== بررسی شناسه رکوردها ===")

record_ids = {}

for split_name, split_records in records.items():
    ids = {
        item["id"]
        for item in split_records
        if "id" in item
    }
    record_ids[split_name] = ids
    print(f"{split_name}: {len(ids)} شناسه")

for i in range(len(SPLITS)):
    for j in range(i + 1, len(SPLITS)):
        first = SPLITS[i]
        second = SPLITS[j]
        overlap = record_ids[first] & record_ids[second]

        print(
            f"هم‌پوشانی {first} و {second}: "
            f"{len(overlap)} شناسه مشترک"
        )

# بررسی شناسه متن‌های انسانی مرتبط
print("\n=== بررسی human_id ===")

human_ids = {}

for split_name, split_records in records.items():
    ids = {
        item["human_id"]
        for item in split_records
        if "human_id" in item
    }
    human_ids[split_name] = ids
    print(f"{split_name}: {len(ids)} مقدار human_id یکتا")

for i in range(len(SPLITS)):
    for j in range(i + 1, len(SPLITS)):
        first = SPLITS[i]
        second = SPLITS[j]
        overlap = human_ids[first] & human_ids[second]

        print(
            f"هم‌پوشانی human_id بین {first} و {second}: "
            f"{len(overlap)} مقدار مشترک"
        )
