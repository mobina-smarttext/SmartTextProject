import re
from io import BytesIO

import pymupdf
from docx import Document


def extract_text_from_txt(uploaded_file):
    """استخراج متن از فایل TXT."""

    file_bytes = uploaded_file.getvalue()

    try:
        return file_bytes.decode("utf-8-sig")

    except UnicodeDecodeError:
        raise ValueError(
            "رمزگذاری فایل پشتیبانی نمی‌شود. "
            "لطفاً فایل TXT را با UTF-8 ذخیره کنید."
        )


def extract_text_from_pdf(uploaded_file):
    """استخراج و مرتب‌سازی اولیه متن از PDF متنی."""

    file_bytes = uploaded_file.getvalue()

    try:
        with pymupdf.open(
            stream=file_bytes,
            filetype="pdf"
        ) as document:

            pages = []

            for page in document:
                page_text = page.get_text("text")

                # یکسان‌سازی فاصله‌های اضافی
                page_text = page_text.replace("\r\n", "\n")
                page_text = page_text.replace("\r", "\n")

                # اتصال خطوط شکسته‌شده با فاصله
                page_text = re.sub(
                    r"(?<=\S)\n(?=\S)",
                    " ",
                    page_text
                )

                # حذف فاصله‌های تکراری
                page_text = re.sub(
                    r"[ \t]+",
                    " ",
                    page_text
                )

                pages.append(page_text.strip())

        return "\n\n".join(pages)

    except Exception as error:
        raise ValueError(
            "خواندن فایل PDF امکان‌پذیر نیست. "
            "ممکن است فایل خراب یا رمزگذاری‌شده باشد."
        ) from error



def extract_text_from_docx(uploaded_file):
    """استخراج متن از فایل Word."""

    file_bytes = uploaded_file.getvalue()

    try:
        document = Document(BytesIO(file_bytes))

        paragraphs = [
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        return "\n".join(paragraphs)

    except Exception as error:
        raise ValueError(
            "خواندن فایل Word امکان‌پذیر نیست. "
            "لطفاً یک فایل DOCX معتبر انتخاب کنید."
        ) from error


def extract_text_from_file(uploaded_file):
    """انتخاب روش استخراج بر اساس نوع فایل."""

    extension = uploaded_file.name.lower().rsplit(".", 1)[-1]

    if extension == "txt":
        return extract_text_from_txt(uploaded_file)

    if extension == "pdf":
        return extract_text_from_pdf(uploaded_file)

    if extension == "docx":
        return extract_text_from_docx(uploaded_file)

    raise ValueError(
        "نوع فایل پشتیبانی نمی‌شود. "
        "فقط TXT، PDF و DOCX مجاز هستند."
    )
