
import unittest

from modules.ai_detector.segment_analyzer import (
    analyze_segments
)


class TestSegmentIntegration(unittest.TestCase):

    def check_real_prediction(self, text, language):
        result = analyze_segments(text)

        self.assertEqual(result["language"], language)
        self.assertEqual(result["total_segments"], 1)

        segment = result["segments"][0]

        self.assertEqual(segment["status"], "analyzed")
        self.assertGreaterEqual(segment["word_count"], 20)

        ai = segment["ai_probability"]
        human = segment["human_probability"]

        self.assertIsNotNone(ai)
        self.assertIsNotNone(human)

        self.assertGreaterEqual(ai, 0.0)
        self.assertLessEqual(ai, 1.0)

        self.assertGreaterEqual(human, 0.0)
        self.assertLessEqual(human, 1.0)

        self.assertAlmostEqual(ai + human, 1.0, places=6)

        self.assertIn(segment["prediction"], [0, 1])

        print(
            f"\n{language} real-model test:"
            f"\nAI score: {ai * 100:.2f}%"
            f"\nHuman score: {human * 100:.2f}%"
            f"\nPrediction: {segment['prediction']}"
        )

    def test_english_real_model(self):
        text = (
            "Artificial intelligence is transforming education "
            "and healthcare in many countries. Researchers use "
            "machine learning to analyze large datasets and "
            "develop new technologies that support decision "
            "making in complex environments."
        )

        self.check_real_prediction(text, "English")

    def test_persian_real_model(self):
        text = (
            "هوش مصنوعی در سال‌های اخیر پیشرفت زیادی داشته است "
            "و در زمینه‌های مختلفی مانند آموزش، پزشکی و صنعت "
            "کاربرد دارد. پژوهشگران از روش‌های یادگیری ماشین "
            "برای تحلیل اطلاعات و شناسایی الگوهای پیچیده "
            "استفاده می‌کنند و تلاش دارند عملکرد سامانه‌های "
            "هوشمند را بهبود دهند."
        )

        self.check_real_prediction(text, "Persian")

    def test_short_text_without_prediction(self):
        text = "This is a short paragraph."

        result = analyze_segments(text)
        segment = result["segments"][0]

        self.assertEqual(segment["status"], "too_short")
        self.assertIsNone(segment["ai_probability"])
        self.assertIsNone(segment["prediction"])


if __name__ == "__main__":
    unittest.main()
