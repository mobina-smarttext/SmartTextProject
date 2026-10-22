
import json
import re
import unicodedata
from pathlib import Path
from collections import defaultdict, Counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

SPLITS = ["train", "val", "test"]


def normalize_text(text):
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("ي", "ی").replace("ك", "ک")
    text = re.sub(r"\s+", " ", text)

    return text.strip().casefold()


def load_records(split):
    path = DATA_DIR / f"{split}.jsonl"
    records = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def main():

    datasets = {
        split: load_records(split)
        for split in SPLITS
    }

    groups = defaultdict(list)

    for split, records in datasets.items():
        for record in records:

            human_id = record.get("human_id")

            if human_id is None:
                continue

            groups[str(human_id)].append(
                (split, record)
            )

    print("\n" + "=" * 65)
    print("PERSIAN DATASET GROUP AUDIT")
    print("=" * 65)

    print(f"Total unique human_id groups: {len(groups)}")

    incomplete_groups = []
    cross_split_groups = []

    for human_id, items in groups.items():

        labels = Counter(
            record.get("label")
            for _, record in items
        )

        splits = {
            split
            for split, _ in items
        }

        if (
            len(items) != 4
            or labels.get(0, 0) != 1
            or labels.get(1, 0) != 3
        ):
            incomplete_groups.append(
                (human_id, len(items), dict(labels))
            )

        if len(splits) > 1:
            cross_split_groups.append(
                (human_id, sorted(splits))
            )

    print(
        f"Groups with unexpected structure: "
        f"{len(incomplete_groups)}"
    )

    for item in incomplete_groups[:20]:
        print("  ", item)

    print(
        f"\nGroups crossing splits: "
        f"{len(cross_split_groups)}"
    )

    for item in cross_split_groups[:20]:
        print("  ", item)

    print("\n--- Duplicate human texts across groups ---")

    human_text_groups = defaultdict(list)

    for human_id, items in groups.items():

        for split, record in items:

            if record.get("label") != 0:
                continue

            normalized = normalize_text(
                record.get("text", "")
            )

            if normalized:
                human_text_groups[normalized].append(
                    (human_id, split)
                )

    duplicate_human_groups = [
        items
        for items in human_text_groups.values()
        if len(set(hid for hid, _ in items)) > 1
    ]

    print(
        "Duplicate human-text groups: "
        f"{len(duplicate_human_groups)}"
    )

    for index, items in enumerate(
        duplicate_human_groups,
        start=1
    ):

        print(
            f"Duplicate group #{index}: {items}"
        )

    print("\nAudit completed.")


if __name__ == "__main__":
    main()
