import csv
import io
import os
from datetime import timezone
from zoneinfo import ZoneInfo

import psycopg2


ISRAEL_TZ = ZoneInfo("Asia/Jerusalem")


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chat_logs (
    id BIGSERIAL PRIMARY KEY,
    asked_at TIMESTAMPTZ NOT NULL,
    conversation_id VARCHAR(64),
    question TEXT NOT NULL,
    answer TEXT,
    response_time_seconds DOUBLE PRECISION,
    sources TEXT,
    source_count INTEGER NOT NULL DEFAULT 0,
    status VARCHAR(32) NOT NULL,
    error TEXT
)
"""


def _database_url():
    return os.environ.get("DATABASE_URL", "").strip()


def log_interaction(
    asked_at,
    conversation_id,
    question,
    answer,
    response_time_seconds,
    sources,
    status,
    error="",
):
    """Persist one chatbot interaction without ever breaking the chatbot.

    Logging is intentionally best-effort. If DATABASE_URL is not configured or
    Postgres is temporarily unavailable, the chatbot continues normally.
    """
    database_url = _database_url()
    if not database_url:
        print("CHAT LOGGING: DATABASE_URL is not configured; skipping log entry.")
        return False

    source_list = list(sources or [])
    try:
        with psycopg2.connect(database_url, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLE_SQL)
                cur.execute(
                    """
                    INSERT INTO chat_logs (
                        asked_at,
                        conversation_id,
                        question,
                        answer,
                        response_time_seconds,
                        sources,
                        source_count,
                        status,
                        error
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        asked_at,
                        conversation_id,
                        question,
                        answer,
                        float(response_time_seconds),
                        "\n".join(source_list),
                        len(source_list),
                        status,
                        error or "",
                    ),
                )
        return True
    except Exception as exc:
        print(f"CHAT LOGGING ERROR: {exc}")
        return False


def export_logs_csv():
    """Return all stored logs as an Excel-friendly UTF-8 CSV byte stream."""
    database_url = _database_url()
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured")

    with psycopg2.connect(database_url, connect_timeout=5) as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
            cur.execute(
                """
                SELECT
                    asked_at,
                    conversation_id,
                    question,
                    answer,
                    response_time_seconds,
                    sources,
                    source_count,
                    status,
                    error
                FROM chat_logs
                ORDER BY asked_at ASC, id ASC
                """
            )
            rows = cur.fetchall()

    text_buffer = io.StringIO(newline="")
    writer = csv.writer(text_buffer)
    writer.writerow([
        "Date",
        "Time",
        "Timezone",
        "Question",
        "Response Time (sec)",
        "Status",
        "Answer",
        "Sources",
        "Source Count",
        "Error",
        "Conversation ID",
    ])

    for row in rows:
        asked_at = row[0]
        if asked_at.tzinfo is None:
            asked_at = asked_at.replace(tzinfo=timezone.utc)
        local_time = asked_at.astimezone(ISRAEL_TZ)

        writer.writerow([
            local_time.strftime("%Y-%m-%d"),
            local_time.strftime("%H:%M:%S"),
            "Asia/Jerusalem",
            row[2],
            round(float(row[4] or 0.0), 3),
            row[7],
            row[3] or "",
            row[5] or "",
            row[6] or 0,
            row[8] or "",
            row[1] or "",
        ])

    # UTF-8 BOM makes Hebrew open correctly when the CSV is opened in Excel.
    return ("\ufeff" + text_buffer.getvalue()).encode("utf-8")
