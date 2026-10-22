
import re


def detect_language(text):
    """
    Detect the dominant language of a text.

    Returns:
        language: Persian, English, or Unknown
        mixed_language: True or False
    """

    if not isinstance(text, str):
        return "Unknown", False

    persian_letters = len(
        re.findall(r"[\u0600-\u06FF]", text)
    )

    english_letters = len(
        re.findall(r"[A-Za-z]", text)
    )

    total = persian_letters + english_letters

    if total == 0:
        return "Unknown", False

    persian_ratio = persian_letters / total
    english_ratio = english_letters / total

    mixed_language = (
        persian_ratio >= 0.20
        and english_ratio >= 0.20
    )

    if persian_letters >= english_letters:
        return "Persian", mixed_language

    return "English", mixed_language
