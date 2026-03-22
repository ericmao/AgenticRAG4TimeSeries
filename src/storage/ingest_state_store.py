"""Wazuh Indexer 輪詢游標讀寫。"""
from __future__ import annotations

from typing import Any, Optional

from src.storage.analysis_runs_store import ensure_ingest_state_schema


def get_state(database_url: str, key_name: str) -> dict[str, Any]:
    import psycopg2
    from psycopg2.extras import RealDictCursor

    ensure_ingest_state_schema(database_url)
    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT key_name, last_timestamp_ms, last_pit_id, updated_at FROM wazuh_ingest_state WHERE key_name = %s",
                (key_name,),
            )
            row = cur.fetchone()
            return dict(row) if row else {}
    finally:
        conn.close()


def upsert_state(
    database_url: str,
    key_name: str,
    *,
    last_timestamp_ms: Optional[int] = None,
    last_pit_id: Optional[str] = None,
) -> None:
    import psycopg2

    ensure_ingest_state_schema(database_url)
    conn = psycopg2.connect(database_url)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO wazuh_ingest_state (key_name, last_timestamp_ms, last_pit_id, updated_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (key_name) DO UPDATE SET
                    last_timestamp_ms = COALESCE(EXCLUDED.last_timestamp_ms, wazuh_ingest_state.last_timestamp_ms),
                    last_pit_id = COALESCE(EXCLUDED.last_pit_id, wazuh_ingest_state.last_pit_id),
                    updated_at = NOW()
                """,
                (key_name, last_timestamp_ms, last_pit_id),
            )
    finally:
        conn.close()
