
import sys
import unittest
from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from modules.ai_detector.language_detector import (
    detect_language
)


MODEL_DIR = PROJECT_ROOT / "models"

PERSIAN_MODEL = (
    MODEL_DIR / "ai_detector_svm.joblib"
)

ENGLISH_MODEL = (
    MODEL_DIR / "ai_detector_svm_english.joblib"
)


class TestLanguageDetection(unittest.TestCase):

    def test_persian_text(self):
        text = (
            "هوش مصنوعی در سال‌های اخیر "
            "پیشرفت زیادی داشته است."
        )

        language, mixed = detect_language(text)

        self.assertEqual(language, "Persian")
        self.assertFalse(mixed)

    def test_english_text(self):
        text = (
            "Artificial intelligence has made "
            "significant progress in recent years."
        )

        language, mixed = detect_language(text)

        self.assertEqual(language, "English")
        self.assertFalse(mixed)

    def test_mixed_text(self):
        text = (
            "هوش مصنوعی Artificial Intelligence "
            "در آموزش کاربرد دارد."
        )

        language, mixed = detect_language(text)

        self.assertEqual(language, "Persian")
        self.assertTrue(mixed)

    def test_empty_text(self):
        language, mixed = detect_language("")

        self.assertEqual(language, "Unknown")
        self.assertFalse(mixed)

    def test_numbers_only(self):
        language, mixed = detect_language(
            "12345 67890"
        )

        self.assertEqual(language, "Unknown")
        self.assertFalse(mixed)

    def test_non_string_input(self):
        language, mixed = detect_language(None)

        self.assertEqual(language, "Unknown")
        self.assertFalse(mixed)


class TestBilingualModels(unittest.TestCase):

    def test_persian_model_prediction(self):
        self.assertTrue(PERSIAN_MODEL.exists())

        model = joblib.load(PERSIAN_MODEL)

        text = (
            "هوش مصنوعی در حوزه‌های مختلف "
            "علوم کامپیوتر کاربرد دارد و "
            "می‌تواند داده‌ها را تحلیل کند."
        )

        prediction = model.predict([text])[0]

        self.assertIn(int(prediction), [0, 1])

    def test_english_model_prediction(self):
        self.assertTrue(ENGLISH_MODEL.exists())

        model = joblib.load(ENGLISH_MODEL)

        text = (
            "Artificial intelligence is used "
            "in many areas of computer science "
            "and can help analyze large datasets."
        )

        prediction = model.predict([text])[0]

        self.assertIn(int(prediction), [0, 1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
