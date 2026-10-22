
import json
import re
from pathlib import Path

from model_loader import get_semantic_model
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_FILE = PROJECT_ROOT / "data" / "plagiarism" / "documents.json"

model = get_semantic_model()
# آستانه آزمایشی نمایش شباهت
SIMILARITY_THRESHOLD = 55.0


def split_sentences(text: str) -> list[str]:
    """تقسیم متن به جمله‌ها با استفاده از علائم نگارشی."""

    text = re.sub(r"\s+", " ", text.strip())

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?؟؛])\s+", text)

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def find_similar_sentences(input_text: str) -> list[dict]:
    """پیدا کردن نزدیک‌ترین جمله در هر سند برای هر جمله ورودی."""

    if not input_text.strip():
        raise ValueError("متن ورودی نباید خالی باشد.")

    if not DOCUMENTS_FILE.exists():
        raise FileNotFoundError(
            f"فایل اسناد پیدا نشد: {DOCUMENTS_FILE}"
        )

    with DOCUMENTS_FILE.open("r", encoding="utf-8") as file:
        documents = json.load(file)

    if not isinstance(documents, list):
        raise ValueError("ساختار فایل اسناد معتبر نیست.")

    input_sentences = split_sentences(input_text)
    document_sentences = []

    for document in documents:
        if (
            not isinstance(document, dict)
            or not isinstance(document.get("title"), str)
            or not isinstance(document.get("text"), str)
        ):
            raise ValueError("یکی از اسناد ساختار معتبری ندارد.")

        for sentence in split_sentences(document["text"]):
            document_sentences.append({
                "title": document["title"],
                "text": sentence,
            })

    if not input_sentences or not document_sentences:
        return []

    # تولید بردار معنایی همه جمله‌ها در یک مرحله
    all_sentences = input_sentences + [
        item["text"] for item in document_sentences
    ]

    model = get_semantic_model()

    embeddings = model.encode(
        all_sentences,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    input_embeddings = embeddings[:len(input_sentences)]
    document_embeddings = embeddings[len(input_sentences):]

    similarity_matrix = cosine_similarity(
        input_embeddings,
        document_embeddings,
    )

    results = []

    # بررسی جداگانه هر جمله ورودی در هر سند
    for input_index, input_sentence in enumerate(input_sentences):

        for document in documents:
            title = document["title"]

            # پیدا کردن شماره جمله‌های متعلق به این سند
            indices = [
                index
                for index, item in enumerate(document_sentences)
                if item["title"] == title
            ]

            if not indices:
                continue

            # پیدا کردن نزدیک‌ترین جمله در همین سند
            best_index = max(
                indices,
                key=lambda index: similarity_matrix[
                    input_index, index
                ],
            )

            score = float(
                similarity_matrix[input_index, best_index]
            )

            score_percent = max(
                0.0,
                min(100.0, score * 100),
            )

            if score_percent >= SIMILARITY_THRESHOLD:
                matched = document_sentences[best_index]

                results.append({
                    "input_sentence": input_sentence,
                    "matched_title": title,
                    "matched_sentence": matched["text"],
                    "semantic_similarity_percent": round(
                        score_percent, 2
                    ),
                })

    results.sort(
        key=lambda item: item["semantic_similarity_percent"],
        reverse=True,
    )

    return results


if __name__ == "__main__":
    sample_text = (
        "یادگیری ماشین یکی از شاخه‌های هوش مصنوعی است. "
        "این روش برای تحلیل داده‌ها و پیش‌بینی کاربرد دارد."
    )

    results = find_similar_sentences(sample_text)

    print("\nجمله‌های دارای شباهت معنایی:\n")

    if not results:
        print("جمله‌ای با آستانه فعلی پیدا نشد.")

    for result in results:
        print(f"جمله ورودی: {result['input_sentence']}")
        print(f"سند مشابه: {result['matched_title']}")
        print(f"جمله ذخیره‌شده: {result['matched_sentence']}")
        print(
            f"شباهت معنایی: "
            f"{result['semantic_similarity_percent']}٪"
        )
        print("-" * 50)
