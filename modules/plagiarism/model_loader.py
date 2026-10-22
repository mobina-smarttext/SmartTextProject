
from functools import lru_cache

from sentence_transformers import SentenceTransformer


MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


@lru_cache(maxsize=1)
def get_semantic_model():
    """
    مدل شباهت معنایی را فقط یک بار
    در هر فرایند پایتون بارگذاری می‌کند.
    """
    return SentenceTransformer(MODEL_NAME)
