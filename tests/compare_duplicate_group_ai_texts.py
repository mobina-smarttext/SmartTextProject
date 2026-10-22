
import json
import re
import unicodedata
from pathlib import Path
from difflib import SequenceMatcher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

DUPLICATE_PAIRS = [
    ("0947", "0636"),
    ("0091", "0306"),
    ("0215", "0059"),
    ("0122", "0296"),
]


def normalize_text(text):
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("ي", "ی").replace("ك", "ک")
    text = re.sub(r"\s+", " ", text)

    return text.strip().casefold()


def load_all_records():
    records = []

    for split in ["train", "val", "test"]:

        path = DATA_DIR / f"{split}.jsonl"

        with open(path, "r", encoding="utf-8") as file:
            for line in file:

                if not line.strip():
                    continue

                record = json.loads(line)
                record["_split"] = split
                records.append(record)

    return records


def main():

    records = load_all_records()

    print("\n" + "=" * 65)
    print("AI TEXT SIMILARITY BETWEEN DUPLICATE GROUPS")
    print("=" * 65)

    for first_id, second_id in DUPLICATE_PAIRS:

        first_ai = [
            record
            for record in records
            if str(record.get("human_id")) == first_id
            and record.get("label") == 1
        ]

        second_ai = [
            record
            for record in records
            if str(record.get("human_id")) == second_id
            and record.get("label") == 1
        ]

        print(f"\nGroup {first_id} vs {second_id}")
        print("-" * 45)

        print(f"AI texts in first group: {len(first_ai)}")
        print(f"AI texts in second group: {len(second_ai)}")

        for first in first_ai:

            for second in second_ai:

                first_text = normalize_text(
                    first.get("text", "")
                )

                second_text = normalize_text(
                    second.get("text", "")
                )

                similarity = SequenceMatcher(
                    None,
                    first_text,
                    second_text,
                    autojunk=False
                ).ratio()

                print(
                    f"{first.get('model')} vs "
                    f"{second.get('model')}: "
                    f"{similarity * 100:.2f}%"
                )

    print("\nComparison completed.")


if __name__ == "__main__":
    main()
