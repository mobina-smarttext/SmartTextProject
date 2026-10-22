
import json

from database.db_manager import (
    get_connection,
    initialize_database,
    save_analysis,
    get_analysis_history,
    get_analysis_by_id,
)


def main():
    print("=" * 55)
    print("DATABASE HISTORY TEST")
    print("=" * 55)

    initialize_database()

    test_result = {
        "prediction": 1,
        "ai_probability": 0.91,
        "human_probability": 0.09,
        "text": "PRIVATE TEST TEXT",
        "segments": [
            {
                "segment_number": 1,
                "prediction": 1,
                "ai_probability": 0.91,
                "text": "PRIVATE SEGMENT TEXT",
            }
        ],
    }

    record_id = None

    try:
        record_id = save_analysis(
            analysis_type="ai_detection",
            language="Persian",
            input_filename="test_document.txt",
            text_length=125,
            result=test_result,
        )

        print("\nSaved record ID:", record_id)

        history = get_analysis_history(limit=10)

        assert any(
            row["id"] == record_id
            for row in history
        ), "Saved record not found in history."

        print("History retrieval: PASSED")

        saved_record = get_analysis_by_id(
            record_id
        )

        assert saved_record is not None

        assert (
            saved_record["result"]["prediction"] == 1
        )

        assert (
            saved_record["result"]["segments"][0]
            ["prediction"] == 1
        )

        serialized_result = json.dumps(
            saved_record["result"],
            ensure_ascii=False,
        )

        assert "PRIVATE TEST TEXT" not in serialized_result
        assert "PRIVATE SEGMENT TEXT" not in serialized_result

        print("Single record retrieval: PASSED")
        print("Sensitive text removal: PASSED")

        print("\nSaved sanitized result:")
        print(
            json.dumps(
                saved_record["result"],
                ensure_ascii=False,
                indent=2,
            )
        )

    finally:
        if record_id is not None:
            with get_connection() as connection:
                connection.execute(
                    """
                    DELETE FROM analysis_history
                    WHERE id = ?
                    """,
                    (record_id,),
                )
                connection.commit()

            print("\nTest record cleaned up.")

    print("\nALL DATABASE TESTS PASSED")


if __name__ == "__main__":
    main()
