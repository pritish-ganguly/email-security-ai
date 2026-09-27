"""
SQLite persistence for Email Security AI scan history.

The database stores completed inference results so the dashboard
can display previous email scans.

This module does not perform ML inference.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATABASE_PATH = Path("data/email_security.db")


def _connect() -> sqlite3.Connection:
    """Create a SQLite connection."""

    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


def initialize_database() -> None:
    """Create the scan-history table if it does not exist."""

    with _connect() as connection:

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS scan_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                scanned_at TEXT NOT NULL,

                file_name TEXT NOT NULL,

                sender TEXT,
                receiver TEXT,
                subject TEXT,

                prediction TEXT NOT NULL,
                is_spam INTEGER NOT NULL,

                decision_score REAL NOT NULL,
                decision_threshold REAL NOT NULL,

                risk_score INTEGER NOT NULL,
                risk_level TEXT NOT NULL,

                url_count INTEGER NOT NULL DEFAULT 0,
                ip_url_count INTEGER NOT NULL DEFAULT 0,
                suspicious_keyword_count INTEGER NOT NULL DEFAULT 0,
                attachment_count INTEGER NOT NULL DEFAULT 0,

                html_present INTEGER NOT NULL DEFAULT 0,
                script_present INTEGER NOT NULL DEFAULT 0,

                recommended_action TEXT,

                reasons TEXT
            )
            """
        )

        connection.commit()


def _get_value(
    value: Any,
    name: str,
    default: Any = None,
) -> Any:
    """Read a value from either a dictionary or an object."""

    if isinstance(value, dict):
        return value.get(
            name,
            default,
        )

    return getattr(
        value,
        name,
        default,
    )


def _get_email_value(
    email_data: object,
    field: str,
    default: Any = "",
) -> Any:
    """Safely read parsed email information."""

    if isinstance(email_data, dict):
        return email_data.get(
            field,
            default,
        )

    return getattr(
        email_data,
        field,
        default,
    )


def save_scan(
    result: object,
) -> int:
    """
    Save a completed EmailScanResult.

    Returns:
        Database ID assigned to the scan.
    """

    initialize_database()

    email_data = result.email_data
    ml_result = result.ml_result
    security = result.security_analysis
    risk = result.risk_assessment

    reasons = _get_value(
        risk,
        "reasons",
        [],
    )

    if reasons is None:
        reasons = []

    reasons_text = "\n".join(
        str(reason)
        for reason in reasons
    )

    scanned_at = datetime.now(
        timezone.utc
    ).isoformat()

    with _connect() as connection:

        cursor = connection.execute(
            """
            INSERT INTO scan_history (
                scanned_at,
                file_name,
                sender,
                receiver,
                subject,
                prediction,
                is_spam,
                decision_score,
                decision_threshold,
                risk_score,
                risk_level,
                url_count,
                ip_url_count,
                suspicious_keyword_count,
                attachment_count,
                html_present,
                script_present,
                recommended_action,
                reasons
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                scanned_at,

                str(
                    Path(
                        result.file_path
                    ).name
                ),

                str(
                    _get_email_value(
                        email_data,
                        "sender",
                        "",
                    )
                    or ""
                ),

                str(
                    _get_email_value(
                        email_data,
                        "receiver",
                        "",
                    )
                    or ""
                ),

                str(
                    _get_email_value(
                        email_data,
                        "subject",
                        "",
                    )
                    or ""
                ),

                str(
                    ml_result["label"]
                ),

                int(
                    bool(
                        ml_result["spam"]
                    )
                ),

                float(
                    ml_result["decision_score"]
                ),

                float(
                    ml_result["threshold"]
                ),

                int(
                    _get_value(
                        risk,
                        "risk_score",
                        0,
                    )
                ),

                str(
                    _get_value(
                        risk,
                        "risk_level",
                        "UNKNOWN",
                    )
                ),

                int(
                    _get_value(
                        security,
                        "url_count",
                        0,
                    )
                ),

                int(
                    _get_value(
                        security,
                        "ip_url_count",
                        0,
                    )
                ),

                int(
                    _get_value(
                        security,
                        "suspicious_keyword_count",
                        0,
                    )
                ),

                int(
                    _get_value(
                        security,
                        "attachment_count",
                        0,
                    )
                ),

                int(
                    bool(
                        _get_value(
                            security,
                            "has_html",
                            False,
                        )
                    )
                ),

                int(
                    bool(
                        _get_value(
                            security,
                            "has_script",
                            False,
                        )
                    )
                ),

                str(
                    _get_value(
                        risk,
                        "recommended_action",
                        "",
                    )
                    or ""
                ),

                reasons_text,
            ),
        )

        connection.commit()

        return int(
            cursor.lastrowid
        )


def get_recent_scans(
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return the most recent scan records."""

    initialize_database()

    limit = max(
        1,
        min(
            int(limit),
            500,
        ),
    )

    with _connect() as connection:

        rows = connection.execute(
            """
            SELECT
                id,
                scanned_at,
                file_name,
                sender,
                receiver,
                subject,
                prediction,
                is_spam,
                decision_score,
                decision_threshold,
                risk_score,
                risk_level,
                url_count,
                ip_url_count,
                suspicious_keyword_count,
                attachment_count,
                html_present,
                script_present,
                recommended_action,
                reasons
            FROM scan_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [
        dict(row)
        for row in rows
    ]


def get_scan_count() -> int:
    """Return the total number of stored scans."""

    initialize_database()

    with _connect() as connection:

        row = connection.execute(
            """
            SELECT COUNT(*) AS total
            FROM scan_history
            """
        ).fetchone()

    return int(
        row["total"]
    )


def clear_scan_history() -> None:
    """Delete all stored scan history."""

    initialize_database()

    with _connect() as connection:

        connection.execute(
            "DELETE FROM scan_history"
        )

        connection.commit()
