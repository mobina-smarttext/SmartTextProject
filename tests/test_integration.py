
import sys
import unittest
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "modules" / "plagiarism")
)

from combined_similarity import compare_documents
from sentence_matching import find_similar_sentences


class TestIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        model_path = (
            PROJECT_ROOT / "models" / "ai_detector_svm.joblib"
        )
        cls.ai_model = joblib.load(model_path)

    def test_ai_model_prediction(self):
        text = "هوش مصنوعی در آموزش کاربرد دارد."

        prediction = self.ai_model.predict([text])[0]

        self.assertIn(int(prediction), [0, 1])

    def test_similarity_documents(self):
        text = (
            "هوش مصنوعی در سال‌های اخیر "
            "پیشرفت زیادی داشته است."
        )

        results = compare_documents(text)

        self.assertEqual(len(results), 6)

        for result in results:
            self.assertIn(
                "combined_similarity_percent",
                result
            )

    def test_similarity_ranking(self):
        text = (
            "هوش مصنوعی در سال‌های اخیر "
            "پیشرفت زیادی داشته است."
        )

        results = compare_documents(text)

        scores = [
            item["combined_similarity_percent"]
            for item in results
        ]

        self.assertEqual(
            scores,
            sorted(scores, reverse=True)
        )

    def test_sentence_matching(self):
        text = (
            "هوش مصنوعی در سال‌های اخیر "
            "پیشرفت زیادی داشته است."
        )

        matches = find_similar_sentences(text)

        self.assertTrue(
            any(
                item["matched_title"] == "متن اول"
                and item["semantic_similarity_percent"] >= 99
                for item in matches
            )
        )


if __name__ == "__main__":
    unittest.main()
