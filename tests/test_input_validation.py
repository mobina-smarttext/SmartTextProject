
import unittest
from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT / "models" / "ai_detector_svm.joblib"
)


class TestInputValidation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load(MODEL_PATH)

    def test_empty_text(self):
        text = ""

        self.assertFalse(text.strip())

    def test_whitespace_only(self):
        text = "     \n   \t  "

        self.assertFalse(text.strip())

    def test_short_text_prediction(self):
        text = "سلام"

        prediction = self.model.predict([text])[0]

        self.assertIn(int(prediction), [0, 1])

    def test_normal_text_prediction(self):
        text = (
            "هوش مصنوعی در سال‌های اخیر "
            "پیشرفت زیادی داشته است."
        )

        prediction = self.model.predict([text])[0]

        self.assertIn(int(prediction), [0, 1])


if __name__ == "__main__":
    unittest.main()
