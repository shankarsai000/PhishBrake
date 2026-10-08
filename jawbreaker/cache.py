from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path


CACHE_DB_PATH = Path(__file__).resolve().parent.parent / "models" / "threat_cache.db"

_OTP_PATTERN = re.compile(
    r"\b((?:otp|one[- ]time (?:password|passcode|code)|verification code|security code|login code|passcode)"
    r"\s*(?:is|:|=|#)?\s*)\d{4,8}\b",
    re.IGNORECASE,
)
_MONEY_PATTERN = re.compile(
    r"(?<!\w)(?:[$€£¥₹]\s?\d[\d,]*(?:\.\d{1,2})?|"
    r"\b(?:usd|eur|gbp|cad|aud)\s*\d[\d,]*(?:\.\d{1,2})?|"
    r"\d[\d,]*(?:\.\d{1,2})?\s*(?:usd|eur|gbp|cad|aud|dollars?|euros?|pounds?)\b)(?!\w)",
    re.IGNORECASE,
)
_PHONE_PATTERN = re.compile(r"(?<!\w)\+?[\d().\-\s]{8,}\d(?!\w)")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize variable codes, amounts, and phone numbers for pattern matching."""
    normalized = text.lower().strip()
    normalized = _OTP_PATTERN.sub(r"\1<code>", normalized)
    normalized = _MONEY_PATTERN.sub(" <amount> ", normalized)
    normalized = _PHONE_PATTERN.sub(" <phone> ", normalized)
    return _WHITESPACE_PATTERN.sub(" ", normalized).strip()


def hash_pattern(normalized_text: str) -> str:
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


def _open_cache() -> sqlite3.Connection:
    CACHE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(CACHE_DB_PATH, timeout=5)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS threat_cache (
            pattern_hash TEXT PRIMARY KEY,
            normalized_text TEXT NOT NULL,
            analysis_json TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    return connection


def lookup_cache(text: str) -> dict | None:
    normalized = normalize_text(text)
    if not normalized:
        return None
    pattern_hash = hash_pattern(normalized)
    connection = _open_cache()
    try:
        row = connection.execute(
            "SELECT analysis_json FROM threat_cache WHERE pattern_hash = ?",
            (pattern_hash,),
        ).fetchone()
        if row is None:
            return None
        analysis = json.loads(row[0])
        return analysis if isinstance(analysis, dict) else None
    except (json.JSONDecodeError, sqlite3.DatabaseError):
        return None
    finally:
        connection.close()


def save_cache(text: str, analysis: dict) -> None:
    normalized = normalize_text(text)
    if not normalized:
        return
    if not isinstance(analysis, dict):
        raise TypeError("analysis must be a dictionary")

    pattern_hash = hash_pattern(normalized)
    analysis_json = json.dumps(analysis, ensure_ascii=True, sort_keys=True)
    connection = _open_cache()
    try:
        connection.execute(
            """
            INSERT INTO threat_cache (pattern_hash, normalized_text, analysis_json, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(pattern_hash) DO UPDATE SET
                normalized_text = excluded.normalized_text,
                analysis_json = excluded.analysis_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (pattern_hash, normalized, analysis_json),
        )
        connection.commit()
    finally:
        connection.close()
