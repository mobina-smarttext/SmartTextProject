
import json
from pathlib import Path
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"

TARGET_IDS = [
    "0122",
    "0296",
    "0947",
    "0636",
    "0091",
    "0306",
    "0215",
    "0059",
]


def load_records(path):

    records = []

    with open(path, "r", encoding="utf-8") as file:

        for line in file:

            if line.strip():
                records.append(json.loads(line))

    return records


def main():

    datasets = {
        "train": load_records(DATA_DIR / "train.jsonl"),
        "val": load_records(DATA_DIR / "val.jsonl"),
        "test": load_records(DATA_DIR / "test.jsonl"),
    }

    print("\n" + "=" * 65)
    print("PERSIAN RELATED GROUP INSPECTION")
    print("=" * 65)

    for human_id in TARGET_IDS:

        print(f"\nHuman ID: {human_id}")
        print("-" * 45)

        for split, records in datasets.items():

            related = [
                record
                for record in records
                if str(record.get("human_id")) == human_id
            ]

            if not related:
                continue

            labels = Counter(
                record.get("label")
                for record in related
            )

            print(f"Split: {split}")
            print(f"Total records: {len(related)}")
            print(f"Human records: {labels.get(0, 0)}")
            print(f"AI records: {labels.get(1, 0)}")

            for record in related:

                print(
                    "  ID:",
                    record.get("id"),
                    "| Label:",
                    record.get("label"),
                    "| Source:",
                    record.get("source"),
                    "| Model:",
                    record.get("model")
                )

        print()

    print("Inspection completed.")


if __name__ == "__main__":
    main()
