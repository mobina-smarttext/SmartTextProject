
import sys
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pymupdf
from docx import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "modules" / "file_handler")
)

from text_extractor import (
    extract_text_from_txt,
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_file
)


def make_uploaded_file(filename, content):
    """شبیه‌سازی فایل بارگذاری‌شده در Streamlit."""

    return SimpleNamespace(
        name=filename,
        getvalue=lambda: content
    )


class TestTextExtractor(unittest.TestCase):

    def test_persian_text(self):
        text = "این یک متن فارسی آزمایشی است."

        uploaded_file = BytesIO(text.encode("utf-8"))

        result = extract_text_from_txt(uploaded_file)

        self.assertEqual(result, text)

    def test_utf8_bom(self):
        text = "متن فارسی با BOM"

        uploaded_file = BytesIO(text.encode("utf-8-sig"))

        result = extract_text_from_txt(uploaded_file)

        self.assertEqual(result, text)

    def test_empty_file(self):
        uploaded_file = BytesIO(b"")

        result = extract_text_from_txt(uploaded_file)

        self.assertEqual(result, "")

    def test_invalid_encoding(self):
        uploaded_file = BytesIO(b"\xff\xfe\x00")

        with self.assertRaises(ValueError):
            extract_text_from_txt(uploaded_file)

    def test_pdf_extraction(self):
        """آزمایش استخراج متن از PDF."""

        pdf = pymupdf.open()
        page = pdf.new_page()

        expected_text = "This is a PDF test document."

        page.insert_text(
            (72, 72),
            expected_text
        )

        pdf_bytes = pdf.tobytes()
        pdf.close()

        uploaded_file = BytesIO(pdf_bytes)

        result = extract_text_from_pdf(uploaded_file)

        self.assertIn(expected_text, result)

    def test_docx_extraction(self):
        """آزمایش استخراج متن از Word."""

        expected_text = "This is a Word test document."

        document = Document()
        document.add_paragraph(expected_text)

        file_buffer = BytesIO()
        document.save(file_buffer)

        uploaded_file = BytesIO(file_buffer.getvalue())

        result = extract_text_from_docx(uploaded_file)

        self.assertIn(expected_text, result)

    def test_file_type_detection(self):
        """بررسی تشخیص نوع فایل."""

        text = "متن آزمایشی برای تشخیص نوع فایل"

        uploaded_file = make_uploaded_file(
            "sample.txt",
            text.encode("utf-8")
        )

        result = extract_text_from_file(uploaded_file)

        self.assertEqual(result, text)

    def test_unsupported_extension(self):
        """بررسی رد کردن نوع فایل نامعتبر."""

        uploaded_file = make_uploaded_file(
            "sample.exe",
            b"test"
        )

        with self.assertRaises(ValueError):
            extract_text_from_file(uploaded_file)


if __name__ == "__main__":
    unittest.main()
