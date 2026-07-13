"""SQLite persistence layer for Provenance Guard.

Two tables:
  - contents: current state of each submitted piece of content.
  - audit_log: append-only event log (classification + appeal events).

Uses stdlib sqlite3 only, one connection per call via a context manager.
This is simplest and safe under Flask's default threaded dev server since
no connection is shared across requests.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "provenance.db"


@contextmanager
def _connect():
    """Yield a SQLite connection, committing on success and always closing."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _now_iso() -> str:
    """Return current UTC time as an ISO-8601 string with a 'Z' suffix."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def init_db() -> None:
    """Create the contents and audit_log tables if they do not exist."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS contents (
                content_id TEXT PRIMARY KEY,
                creator_id TEXT NOT NULL,
                text TEXT NOT NULL,
                attribution TEXT,
                confidence REAL,
                llm_score REAL,
                stylo_score REAL,
                stylo_metrics TEXT,
                llm_rationale TEXT,
                label TEXT,
                status TEXT NOT NULL DEFAULT 'classified',
                appeal_reasoning TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event TEXT NOT NULL,
                content_id TEXT,
                entry TEXT NOT NULL
            )
            """
        )


def save_submission(record: dict) -> None:
    """Insert a new row into contents.

    `record` is expected to contain: content_id, creator_id, text,
    attribution, confidence, llm_score, stylo_score, stylo_metrics (dict),
    llm_rationale, label, status. Timestamps are generated here.
    """
    now = _now_iso()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO contents (
                content_id, creator_id, text, attribution, confidence,
                llm_score, stylo_score, stylo_metrics, llm_rationale,
                label, status, appeal_reasoning, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["content_id"],
                record["creator_id"],
                record["text"],
                record.get("attribution"),
                record.get("confidence"),
                record.get("llm_score"),
                record.get("stylo_score"),
                json.dumps(record.get("stylo_metrics")),
                record.get("llm_rationale"),
                record.get("label"),
                record.get("status", "classified"),
                None,
                now,
                now,
            ),
        )


def get_content(content_id: str) -> dict | None:
    """Fetch a content record by id, with stylo_metrics JSON-decoded.

    Returns None if no such content_id exists.
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM contents WHERE content_id = ?", (content_id,)
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    if result.get("stylo_metrics"):
        result["stylo_metrics"] = json.loads(result["stylo_metrics"])
    return result


def mark_under_review(content_id: str, reasoning: str) -> None:
    """Flip a content record's status to 'under_review' and store the appeal reasoning."""
    now = _now_iso()
    with _connect() as conn:
        conn.execute(
            """
            UPDATE contents
            SET status = 'under_review', appeal_reasoning = ?, updated_at = ?
            WHERE content_id = ?
            """,
            (reasoning, now, content_id),
        )


def append_log(event: str, content_id: str | None, payload: dict) -> None:
    """Append an event to the audit log. Timestamp is generated here."""
    now = _now_iso()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO audit_log (timestamp, event, content_id, entry) VALUES (?, ?, ?, ?)",
            (now, event, content_id, json.dumps(payload)),
        )


def recent_log(limit: int = 20) -> list[dict]:
    """Return the most recent audit log entries, newest first.

    Each item merges the row metadata with the decoded JSON payload.
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    entries = []
    for row in rows:
        item = {
            "id": row["id"],
            "timestamp": row["timestamp"],
            "event": row["event"],
            "content_id": row["content_id"],
        }
        item.update(json.loads(row["entry"]))
        entries.append(item)
    return entries
