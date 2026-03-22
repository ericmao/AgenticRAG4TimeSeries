"""PostgreSQL：kb_group_llm（每群組意義／使用方向／威脅對應等）。"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

_SCHEMA_ENSURED = False


def ensure_kb_group_llm_schema(database_url: str) -> None:
    global _SCHEMA_ENSURED
    if _SCHEMA_ENSURED:
        return
    import psycopg2

    conn = psycopg2.connect(database_url)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS kb_group_llm (
                    group_key TEXT PRIMARY KEY,
                    meaning TEXT NOT NULL DEFAULT '',
                    usage_direction TEXT NOT NULL DEFAULT '',
                    threats TEXT NOT NULL DEFAULT '',
                    summary TEXT NOT NULL DEFAULT '',
                    content_fingerprint TEXT,
                    model TEXT,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_kb_group_llm_updated
                ON kb_group_llm (updated_at DESC)
                """
            )
        _SCHEMA_ENSURED = True
    finally:
        conn.close()


def reset_schema_cache_for_tests() -> None:
    global _SCHEMA_ENSURED
    _SCHEMA_ENSURED = False


def fetch_all_kb_group_llm(database_url: str) -> dict[str, dict[str, Any]]:
    """group_key -> {meaning, usage_direction, threats, summary, content_fingerprint, model, updated_at}"""
    ensure_kb_group_llm_schema(database_url)
    import psycopg2

    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT group_key, meaning, usage_direction, threats, summary,
                       content_fingerprint, model, updated_at
                FROM kb_group_llm
                """
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    out: dict[str, dict[str, Any]] = {}
    for (
        gk,
        meaning,
        usage_direction,
        threats,
        summary,
        fp,
        model,
        updated_at,
    ) in rows:
        out[gk] = {
            "meaning": (meaning or "").strip(),
            "usage_direction": (usage_direction or "").strip(),
            "threats": (threats or "").strip(),
            "summary": (summary or "").strip(),
            "content_fingerprint": fp,
            "model": model,
            "updated_at": updated_at,
        }
    return out


def upsert_kb_group_llm(
    database_url: str,
    *,
    group_key: str,
    meaning: str,
    usage_direction: str,
    threats: str,
    summary: str = "",
    content_fingerprint: Optional[str] = None,
    model: Optional[str] = None,
) -> None:
    ensure_kb_group_llm_schema(database_url)
    import psycopg2

    now = datetime.now(timezone.utc)
    conn = psycopg2.connect(database_url)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO kb_group_llm (
                    group_key, meaning, usage_direction, threats, summary,
                    content_fingerprint, model, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (group_key) DO UPDATE SET
                    meaning = EXCLUDED.meaning,
                    usage_direction = EXCLUDED.usage_direction,
                    threats = EXCLUDED.threats,
                    summary = EXCLUDED.summary,
                    content_fingerprint = EXCLUDED.content_fingerprint,
                    model = EXCLUDED.model,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    group_key,
                    meaning,
                    usage_direction,
                    threats,
                    summary,
                    content_fingerprint,
                    model,
                    now,
                ),
            )
    finally:
        conn.close()
