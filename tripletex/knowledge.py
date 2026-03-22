"""
SQLite-backed knowledge base for endpoint-specific notes.

One row per endpoint (e.g. "POST /invoice"). The agent reads notes
before planning and writes new rules discovered during execution.

If GCS_BUCKET env var is set, the DB is downloaded from GCS on startup
and uploaded back after each write, so rules persist across deploys.
"""

import logging
import os
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

_GCS_BUCKET = os.getenv("GCS_BUCKET", "")
_GCS_OBJECT = "knowledge.db"

DB_PATH = Path(__file__).parent / "knowledge.db"

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS endpoint_notes (
    endpoint   TEXT PRIMARY KEY,
    notes      TEXT NOT NULL DEFAULT '',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""


def _gcs_download() -> None:
    """Download DB from GCS, overwriting local file."""
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(_GCS_BUCKET)
        blob = bucket.blob(_GCS_OBJECT)
        if blob.exists():
            blob.download_to_filename(str(DB_PATH))
            logger.info("Knowledge: downloaded DB from gs://%s/%s", _GCS_BUCKET, _GCS_OBJECT)
        else:
            logger.info("Knowledge: no DB in GCS yet, starting fresh")
    except Exception as exc:
        logger.warning("Knowledge: GCS download failed: %s", exc)


def _gcs_upload() -> None:
    """Upload local DB to GCS."""
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(_GCS_BUCKET)
        bucket.blob(_GCS_OBJECT).upload_from_filename(str(DB_PATH))
        logger.info("Knowledge: uploaded DB to gs://%s/%s", _GCS_BUCKET, _GCS_OBJECT)
    except Exception as exc:
        logger.warning("Knowledge: GCS upload failed: %s", exc)


def init_db() -> None:
    if _GCS_BUCKET:
        _gcs_download()
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
    if _GCS_BUCKET:
        _gcs_upload()


def get_all_notes() -> dict[str, str]:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT endpoint, notes FROM endpoint_notes").fetchall()
    return {row[0]: row[1] for row in rows}


# Ensure table exists on import
init_db()
