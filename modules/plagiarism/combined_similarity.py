
import json
from pathlib import Path

from model_loader import get_semantic_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_FILE = PROJECT_ROOT / "data" / "plagiarism" / "documents.json"

model = get_semantic_model()
LEXICAL_WEIGHT = 0.5
SEMANTIC_WEIGHT = 0.5


def lexical_similarity(text1: str, text2: str) -> float:
    """محاسبه شباهت واژگانی دو متن."""

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=10000,
    )

    vectors = vectorizer.fit_transform([text1, text2])
    score = cosine_similarity(vectors[0], vectors[1])[0][0]

    return float(score)


def compare_documents(input_text: str) -> list[dict]:
    """مقایسه متن ورودی با اسناد ذخیره‌شده."""

    input_text = input_text.strip()

    if not input_text:
        raise ValueError("متن ورودی نباید خالی باشد.")

    if not DOCUMENTS_FILE.exists():
        raise FileNotFoundError(
            f"فایل اسناد پیدا نشد: {DOCUMENTS_FILE}"
        )

    with DOCUMENTS_FILE.open("r", encoding="utf-8") as file:
        documents = json.load(file)

    if not isinstance(documents, list):
        raise ValueError("ساختار فایل اسناد معتبر نیست.")

    for document in documents:
        if (
            not isinstance(document, dict)
            or not isinstance(document.get("text"), str)
            or not document["text"].strip()
            or "id" not in document
            or "title" not in document
        ):
            raise ValueError(
                "یکی از اسناد ذخیره‌شده ساختار معتبری ندارد."
            )

    if not documents:
        return []

    document_texts = [
        document["text"].strip()
        for document in documents
    ]

    # بارگذاری مدل فقط یک بار انجام می‌شود.
    model = get_semantic_model()

    # بردار متن ورودی و تمام اسناد در یک مرحله ساخته می‌شود.
    all_texts = [input_text] + document_texts

    embeddings = model.encode(
        all_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    input_embedding = embeddings[0:1]
    document_embeddings = embeddings[1:]

    # محاسبه شباهت معنایی همه اسناد به‌صورت یک‌جا
    semantic_scores = cosine_similarity(
        input_embedding,
        document_embeddings,
    )[0]

    results = []

    for index, document in enumerate(documents):
        document_text = document_texts[index]

        lexical_score = lexical_similarity(
            input_text,
            document_text,
        )

        semantic_score = float(semantic_scores[index])
        semantic_score = max(0.0, min(1.0, semantic_score))

        combined_score = (
            LEXICAL_WEIGHT * lexical_score
            + SEMANTIC_WEIGHT * semantic_score
        )

        results.append({
            "id": document["id"],
            "title": document["title"],
            "lexical_similarity_percent": round(
                lexical_score * 100, 2
            ),
            "semantic_similarity_percent": round(
                semantic_score * 100, 2
            ),
            "combined_similarity_percent": round(
                combined_score * 100, 2
            ),
        })

    results.sort(
        key=lambda item: item["combined_similarity_percent"],
        reverse=True,
    )

    return results


if __name__ == "__main__":
    sample_text = (
        "یادگیری ماشین یکی از شاخه‌های هوش مصنوعی است "
        "که برای تحلیل داده‌ها و پیش‌بینی استفاده می‌شود."
    )

    results = compare_documents(sample_text)

    print("\nنتایج مقایسه متن با اسناد ذخیره‌شده:\n")

    for result in results:
        print(f"عنوان: {result['title']}")
        print(
            f"شباهت واژگانی: "
            f"{result['lexical_similarity_percent']}٪"
        )
        print(
            f"شباهت معنایی: "
            f"{result['semantic_similarity_percent']}٪"
        )
        print(
            f"امتیاز ترکیبی آزمایشی: "
            f"{result['combined_similarity_percent']}٪"
        )
        print("-" * 40)
