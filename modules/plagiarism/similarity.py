
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_FILE = PROJECT_ROOT / "data" / "plagiarism" / "documents.json"


def compare_texts(text1: str, text2: str) -> float:
    """محاسبه درصد شباهت واژگانی دو متن."""

    text1 = text1.strip()
    text2 = text2.strip()

    if not text1 or not text2:
        raise ValueError("هر دو متن باید خالی نباشند.")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=10000,
    )

    vectors = vectorizer.fit_transform([text1, text2])
    score = cosine_similarity(vectors[0], vectors[1])[0][0]

    return round(float(score) * 100, 2)


def compare_with_documents(input_text: str) -> list[dict]:
    """مقایسه متن ورودی با تمام اسناد ذخیره‌شده."""

    input_text = input_text.strip()

    if not input_text:
        raise ValueError("متن ورودی نباید خالی باشد.")

    if not DOCUMENTS_FILE.exists():
        raise FileNotFoundError(
            f"فایل اسناد پیدا نشد: {DOCUMENTS_FILE}"
        )

    with DOCUMENTS_FILE.open("r", encoding="utf-8") as file:
        documents = json.load(file)

    results = []

    for document in documents:
        score = compare_texts(input_text, document["text"])

        results.append({
            "id": document["id"],
            "title": document["title"],
            "similarity_percent": score,
        })

    results.sort(
        key=lambda item: item["similarity_percent"],
        reverse=True,
    )

    return results


if __name__ == "__main__":
    sample_text = (
    "یادگیری ماشین یکی از شاخه‌های هوش مصنوعی است "
    "که برای تحلیل داده‌ها و پیش‌بینی استفاده می‌شود."
)
    results = compare_with_documents(sample_text)

    print("\nنتایج مقایسه با اسناد ذخیره‌شده:")
    for result in results:
        print(
            f"{result['title']}: "
            f"{result['similarity_percent']}٪ شباهت"
        )
