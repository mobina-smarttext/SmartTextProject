
import unittest
from unittest.mock import patch

from modules.ai_detector.segment_analyzer import (
    split_into_segments,
    count_words,
    analyze_segments,
)


class TestSegmentAnalyzer(unittest.TestCase):

    def test_empty_text_splitting(self):
        result = split_into_segments("   ")
        self.assertEqual(result, [])

    def test_three_paragraphs(self):
        text = (
            "First paragraph.\n\n"
            "Second paragraph.\n\n"
            "Third paragraph."
        )

        result = split_into_segments(text)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], "First paragraph.")
        self.assertEqual(result[2], "Third paragraph.")

    def test_word_count(self):
        self.assertEqual(
            count_words("This is a simple test."),
            5
        )

    def test_empty_input_rejected(self):
        with self.assertRaises(ValueError):
            analyze_segments("")

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError):
            analyze_segments(None)

    @patch(
        "modules.ai_detector.segment_analyzer."
        "load_calibrated_model"
    )
    def test_short_paragraph_not_analyzed(
        self,
        mock_load_model
    ):
        text = "This paragraph is short."

        result = analyze_segments(text)

        self.assertEqual(result["total_segments"], 1)
        self.assertEqual(
            result["segments"][0]["status"],
            "too_short"
        )
        self.assertIsNone(
            result["segments"][0]["ai_probability"]
        )

    @patch(
        "modules.ai_detector.segment_analyzer."
        "predict_ai_probability"
    )
    @patch(
        "modules.ai_detector.segment_analyzer."
        "load_calibrated_model"
    )
    def test_valid_paragraph_prediction(
        self,
        mock_load_model,
        mock_predict
    ):
        text = " ".join(
            ["Artificial intelligence supports education"]
            * 6
        )

        mock_predict.return_value = {
            "language": "English",
            "mixed_language": False,
            "prediction": 1,
            "ai_probability": 0.85,
            "human_probability": 0.15,
        }

        result = analyze_segments(text)

        segment = result["segments"][0]

        self.assertEqual(
            segment["status"],
            "analyzed"
        )
        self.assertAlmostEqual(
            segment["ai_probability"],
            0.85
        )
        self.assertEqual(
            segment["prediction"],
            1
        )

        mock_predict.assert_called_once()

    @patch(
        "modules.ai_detector.segment_analyzer."
        "predict_ai_probability"
    )
    @patch(
        "modules.ai_detector.segment_analyzer."
        "load_calibrated_model"
    )
    def test_mixed_language_paragraph_skipped(
        self,
        mock_load_model,
        mock_predict
    ):
        english_paragraph = " ".join(
            ["Machine learning supports research"]
            * 6
        )

        mixed_paragraph = (
            "هوش مصنوعی در آموزش کاربرد دارد و "
            "این فناوری در بسیاری از زمینه‌ها "
            "استفاده می‌شود. "
            + " ".join(
                ["Artificial intelligence is useful"]
                * 6
            )
        )

        text = (
            english_paragraph
            + "\n\n"
            + mixed_paragraph
        )

        result = analyze_segments(text)

        self.assertEqual(result["total_segments"], 2)

        self.assertEqual(
            result["segments"][1]["status"],
            "language_mismatch"
        )

        # The mixed-language paragraph must not
        # be passed to the prediction model.
        for call in mock_predict.call_args_list:
            self.assertNotEqual(
                call.args[0],
                mixed_paragraph
            )


if __name__ == "__main__":
    unittest.main()
