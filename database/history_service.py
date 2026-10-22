
from database.db_manager import save_analysis


def save_ai_history(
    language,
    text_length,
    prediction,
    ai_probability,
    human_probability,
    mixed_language,
    segment_result=None,
):
    """
    Save only selected AI detection metadata.

    No original text, filename, or full report
    is passed to the database.
    """

    summary = {
        "prediction": int(prediction),
        "ai_probability": float(ai_probability),
        "human_probability": float(human_probability),
        "mixed_language": bool(mixed_language),
    }

    if segment_result is not None:

        segments = segment_result.get(
            "segments",
            []
        )

        status_counts = {}

        analyzed_count = 0
        ai_count = 0
        human_count = 0

        for segment in segments:

            status = segment.get(
                "status",
                "unknown"
            )

            status_counts[status] = (
                status_counts.get(status, 0) + 1
            )

            if status == "analyzed":

                analyzed_count += 1

                if segment.get("prediction") == 1:
                    ai_count += 1

                elif segment.get("prediction") == 0:
                    human_count += 1

        summary["segments"] = {
            "total": len(segments),
            "analyzed": analyzed_count,
            "predicted_ai": ai_count,
            "predicted_human": human_count,
            "status_counts": status_counts,
        }

    return save_analysis(
        analysis_type="ai_detection",
        language=language,
        input_filename=None,
        text_length=text_length,
        result=summary,
    )


def save_plagiarism_history(
    language,
    text_length,
    document_results,
    sentence_results,
):
    """
    Save only numeric similarity summaries.

    Document titles and matched sentences
    are intentionally excluded.
    """

    scores = []

    for document in document_results:

        scores.append({
            "lexical_similarity_percent": float(
                document["lexical_similarity_percent"]
            ),
            "semantic_similarity_percent": float(
                document["semantic_similarity_percent"]
            ),
            "combined_similarity_percent": float(
                document["combined_similarity_percent"]
            ),
        })

    max_combined_similarity = (
        max(
            item["combined_similarity_percent"]
            for item in scores
        )
        if scores
        else None
    )

    summary = {
        "compared_documents": len(scores),
        "similar_sentence_count": len(
            sentence_results
        ),
        "max_combined_similarity_percent": (
            max_combined_similarity
        ),
        "document_scores": scores,
    }

    return save_analysis(
        analysis_type="plagiarism",
        language=language,
        input_filename=None,
        text_length=text_length,
        result=summary,
    )
