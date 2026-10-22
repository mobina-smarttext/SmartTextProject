
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "modules" / "plagiarism")
)

from sentence_matching import split_sentences


class TestPlagiarism(unittest.TestCase):

    def test_persian_sentence_splitting(self):
        text = "هوش مصنوعی پیشرفت کرده است. یادگیری ماشین کاربرد دارد."

        result = split_sentences(text)

        self.assertEqual(len(result), 2)
        self.assertEqual(
            result[0],
            "هوش مصنوعی پیشرفت کرده است."
        )

    def test_empty_text(self):
        result = split_sentences("   ")

        self.assertEqual(result, [])

    def test_question_mark(self):
        text = "هوش مصنوعی چیست؟ کاربرد آن زیاد است."

        result = split_sentences(text)

        self.assertEqual(len(result), 2)

    def test_extra_spaces(self):
        text = "سلام.    حال شما چطور است؟"

        result = split_sentences(text)

        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
