import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "raw"
FILES = ["train.jsonl", "test.jsonl", "val.jsonl"]


def check_file(file_path: Path) -> None:
    print(f"\n بررسی فایل: {file_path.name}")

    if not file_path.exists():
        print("خطا: فایل پیدا نشد.")
        return

    total = 0
    labels = {}
    errors = 0

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                total += 1

                label = record.get("label", "نامشخص")
                labels[label] = labels.get(label, 0) + 1

                if "text" not in record:
                    print(f"هشدار: فیلد text در خط {line_number} وجود ندارد.")
                    errors += 1

            except json.JSONDecodeError:
                print(f"خطای JSON در خط {line_number}")
                errors += 1

    print(f"تعداد رکوردها: {total}")
    print(f"توزیع برچسب‌ها: {labels}")
    print(f"تعداد خطاها: {errors}")


if __name__ == "__main__":
    for filename in FILES:
        check_file(DATA_DIR / filename)