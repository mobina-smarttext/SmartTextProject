
import re

from modules.ai_detector.language_detector import detect_language
from modules.ai_detector.prediction_service import (
    load_calibrated_model,
    predict_ai_probability,
)


MIN_SEGMENT_WORDS = 20


def split_into_segments(text):
    """
    Split a document into paragraphs.

    Empty paragraphs are ignored.
    """

    if not isinstance(text, str):
        raise ValueError("Text must be a string.")

    text = text.strip()

    if not text:
        return []

    paragraphs = re.split(
        r"\n\s*\n+",
        text
    )

    segments = [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]

    return segments


def count_words(text):
    """
    Count Persian and English words.
    """

    return len(
        re.findall(
            r"[\w\u200c]+",
            text,
            flags=re.UNICODE
        )
    )


def analyze_segments(text):
    """
    Analyze document paragraphs separately.

    Segment scores are exploratory and have
    not been validated as authorship probabilities.
    """

    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            "Input text must not be empty."
        )

    language, mixed_language = detect_language(text)

    if language == "Unknown":
        raise ValueError(
            "Could not detect Persian or English."
        )

    model = load_calibrated_model(language)

    segments = split_into_segments(text)

    results = []

    for index, segment in enumerate(
        segments,
        start=1
    ):

        word_count = count_words(segment)

        if word_count < MIN_SEGMENT_WORDS:

            results.append({
                "segment_number": index,
                "text": segment,
                "word_count": word_count,
                "status": "too_short",
                "ai_probability": None,
                "human_probability": None,
                "prediction": None,
            })

            continue

        segment_language, segment_mixed = (
            detect_language(segment)
        )

        if (
            segment_language != language
            or segment_mixed
        ):

            results.append({
                "segment_number": index,
                "text": segment,
                "word_count": word_count,
                "status": "language_mismatch",
                "ai_probability": None,
                "human_probability": None,
                "prediction": None,
            })

            continue

        prediction_result = predict_ai_probability(
            segment,
            model=model
        )

        results.append({
            "segment_number": index,
            "text": segment,
            "word_count": word_count,
            "status": "analyzed",
            "ai_probability": prediction_result[
                "ai_probability"
            ],
            "human_probability": prediction_result[
                "human_probability"
            ],
            "prediction": prediction_result[
                "prediction"
            ],
        })

    return {
        "language": language,
        "mixed_language": mixed_language,
        "total_segments": len(segments),
        "segments": results,
    }


if __name__ == "__main__":

    sample_text = """
Artificial intelligence is transforming education
and healthcare in many countries. Researchers use
machine learning to analyze large datasets and
develop new technologies that support decision
making in complex environments.

Yesterday I visited my university and discussed
our final project with my professor. We reviewed
the tasks we had completed and decided to test
the software again before preparing the report.

Machine learning systems can recognize patterns
in large collections of information. These
systems are used in medical research, language
processing, and many other fields where
automated analysis can support human experts.
"""

    result = analyze_segments(sample_text)

    print(
        f"Language: {result['language']}"
    )

    print(
        f"Total segments: {result['total_segments']}"
    )

    for segment in result["segments"]:

        print("\n" + "-" * 50)

        print(
            f"Segment: {segment['segment_number']}"
        )

        print(
            f"Words: {segment['word_count']}"
        )

        print(
            f"Status: {segment['status']}"
        )

        if segment["status"] == "analyzed":

            print(
                "AI score: "
                f"{segment['ai_probability'] * 100:.2f}%"
            )

            print(
                "Human score: "
                f"{segment['human_probability'] * 100:.2f}%"
            )

            print(
                f"Prediction: {segment['prediction']}"
            )
