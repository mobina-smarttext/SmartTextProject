
import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "raw"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "processed"

FILES = ["train.jsonl", "val.jsonl", "test.jsonl"]


def normalize_text(text: str) -> str:
    """Normalize Persian and English text without changing the source file."""

    text = str(text)

    # Normalize Arabic characters to Persian equivalents
    text = text.replace("ي", "ی")
    text = text.replace("ى", "ی")
    text = text.replace("ك", "ک")

    # Normalize different forms of whitespace
    text = text.replace("\u200c", "‌")  # Persian half-space
    text = re.sub(r"[ \t\r\n]+", " ", text)

    return text.strip()


def process_file(filename: str) -> None:
    input_path = DATA_DIR / filename
    output_path = OUTPUT_DIR / filename

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    empty_texts = 0

    with input_path.open("r", encoding="utf-8") as source, \
         output_path.open("w", encoding="utf-8") as destination:

        for line in source:
            if not line.strip():
                continue

            record = json.loads(line)
            original_text = record.get("text", "")
            cleaned_text = normalize_text(original_text)

            if not cleaned_text:
                empty_texts += 1

            record["text"] = cleaned_text

            destination.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
            total += 1

    print(f"\nفایل: {filename}")
    print(f"تعداد رکورد پردازش‌شده: {total}")
    print(f"متن خالی: {empty_texts}")
    print(f"خروجی: {output_path}")


if __name__ == "__main__":
    for filename in FILES:
        process_file(filename)

    print("\nپیش‌پردازش اولیه تمام شد.")
