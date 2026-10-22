
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def compare_semantic_similarity(text1: str, text2: str) -> float:
    """محاسبه شباهت معنایی دو متن."""

    if not text1.strip() or not text2.strip():
        raise ValueError("هر دو متن باید خالی نباشند.")

    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode([text1, text2])
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    return round(float(score) * 100, 2)


if __name__ == "__main__":
    text1 = "هوش مصنوعی در آموزش و پزشکی کاربرد دارد."
    text2 = "از فناوری هوش مصنوعی در حوزه‌های درمان و یادگیری استفاده می‌شود."

    score = compare_semantic_similarity(text1, text2)

    print("درصد شباهت معنایی:", score)
