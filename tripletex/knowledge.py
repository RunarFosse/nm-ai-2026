"""
SQLite-backed knowledge base for endpoint-specific notes.

One row per endpoint (e.g. "POST /invoice"). The agent reads notes
before planning and writes new rules discovered during execution
(DEV_MODE only).
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "knowledge.db"

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS endpoint_notes (
    endpoint   TEXT PRIMARY KEY,
    notes      TEXT NOT NULL DEFAULT '',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(_CREATE_SQL)


def get_notes(endpoint: str) -> str | None:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT notes FROM endpoint_notes WHERE endpoint = ?", (endpoint,)
        ).fetchone()
    return row[0] if row else None


def upsert_notes(endpoint: str, notes: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO endpoint_notes (endpoint, notes, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(endpoint) DO UPDATE SET
                notes = excluded.notes,
                updated_at = CURRENT_TIMESTAMP
            """,
            (endpoint, notes),
        )


def get_all_notes() -> dict[str, str]:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT endpoint, notes FROM endpoint_notes").fetchall()
    return {row[0]: row[1] for row in rows}


# Ensure table exists on import
init_db()
