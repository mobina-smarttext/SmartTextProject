
import json
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_PATH = (
    PROJECT_ROOT
    / "database"
    / "smarttext.db"
)

ALLOWED_ANALYSIS_TYPES = {
    "ai_detection",
    "plagiarism",
    "combined",
}

# These keys may contain original student text.
SENSITIVE_KEYS = {
    "text",
    "original_text",
    "input_text",
    "raw_text",
    "content",
    "segment_text",
    "matched_text",
    "source_text",
    "query_text",
    "excerpt",
    "snippet",
}


def get_connection():
    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL
                    DEFAULT CURRENT_TIMESTAMP,
                analysis_type TEXT NOT NULL,
                language TEXT,
                input_filename TEXT,
                text_length INTEGER,
                result_json TEXT NOT NULL
            )
            """
        )

        connection.commit()


def sanitize_result(value):
    """
    Remove known text-bearing fields recursively.

    Only structured JSON-compatible data is supported.
    This is a basic safeguard, not a complete privacy filter.
    """

    if isinstance(value, dict):
        return {
            key: sanitize_result(item)
            for key, item in value.items()
            if str(key).lower() not in SENSITIVE_KEYS
        }

    if isinstance(value, (list, tuple)):
        return [
            sanitize_result(item)
            for item in value
        ]

    if value is None or isinstance(
        value,
        (str, int, float, bool)
    ):
        return value

    raise TypeError(
        f"Unsupported result value type: {type(value)}"
    )


def save_analysis(
    analysis_type,
    result,
    language=None,
    input_filename=None,
    text_length=None,
):
    """
    Save analysis metadata and a sanitized result.

    Returns the database ID of the saved record.
    """

    if analysis_type not in ALLOWED_ANALYSIS_TYPES:
        raise ValueError(
            f"Unsupported analysis type: {analysis_type}"
        )

    if not isinstance(result, dict):
        raise ValueError(
            "Analysis result must be a dictionary."
        )

    if text_length is not None:
        if (
            type(text_length) is not int
            or text_length < 0
        ):
            raise ValueError(
                "text_length must be a non-negative integer."
            )

    safe_result = sanitize_result(result)

    result_json = json.dumps(
        safe_result,
        ensure_ascii=False,
        allow_nan=False,
    )

    initialize_database()

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO analysis_history (
                analysis_type,
                language,
                input_filename,
                text_length,
                result_json
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                analysis_type,
                language,
                input_filename,
                text_length,
                result_json,
            ),
        )

        connection.commit()

        return cursor.lastrowid


def get_analysis_history(limit=50):
    """
    Return recent analyses, newest first.

    Results are returned as dictionaries.
    """

    if type(limit) is not int or limit < 1:
        raise ValueError(
            "limit must be a positive integer."
        )

    initialize_database()

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                created_at,
                analysis_type,
                language,
                input_filename,
                text_length
            FROM analysis_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [
        dict(row)
        for row in rows
    ]


def get_analysis_by_id(analysis_id):
    """
    Return a single analysis with its saved result.
    """

    if (
        type(analysis_id) is not int
        or analysis_id < 1
    ):
        raise ValueError(
            "analysis_id must be a positive integer."
        )

    initialize_database()

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM analysis_history
            WHERE id = ?
            """,
            (analysis_id,),
        ).fetchone()

    if row is None:
        return None

    analysis = dict(row)

    analysis["result"] = json.loads(
        analysis.pop("result_json")
    )

    return analysis


def main():
    print("=" * 55)
    print("SMARTTEXT SQLITE DATABASE")
    print("=" * 55)

    initialize_database()

    print("\nDatabase initialized successfully.")
    print("Database path:", DATABASE_PATH)

    with get_connection() as connection:
        tables = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
            """
        ).fetchall()

        print("\nDatabase tables:")

        for table in tables:
            print("-", table["name"])

        columns = connection.execute(
            """
            PRAGMA table_info(analysis_history)
            """
        ).fetchall()

        print("\nAnalysis history columns:")

        for column in columns:
            print(
                "-",
                column["name"],
                "(" + column["type"] + ")"
            )


if __name__ == "__main__":
    main()
