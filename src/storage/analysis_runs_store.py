"""PostgreSQL：analysis_runs 建立、寫入、查詢。"""
from __future__ import annotations

from typing import Any, Optional

def ensure_analysis_runs_schema(database_url: str) -> None:
    """若表不存在則建立（簡化部署；正式環境建議先跑 SQL 檔）。"""
    import psycopg2

    conn = psycopg2.connect(database_url)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_runs (
                    id SERIAL PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    source TEXT NOT NULL CHECK (source IN ('cert', 'wazuh')),
                    dataset_label TEXT,
                    episode_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    target_ip TEXT,
                    window_start_ms BIGINT NOT NULL,
                    window_end_ms BIGINT NOT NULL,
                    alert_count INT NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'pending',
                    evidence_json JSONB,
                    agent_outputs_json JSONB,
                    issues_json JSONB,
                    error_message TEXT,
                    writeback_json JSONB,
                    layerc_summary JSONB
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_analysis_runs_episode ON analysis_runs(episode_id)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_analysis_runs_source ON analysis_runs(source)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_analysis_runs_created ON analysis_runs(created_at DESC)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_analysis_runs_run_id ON analysis_runs(run_id)"
            )
    finally:
        conn.close()


def ensure_ingest_state_schema(database_url: str) -> None:
    import psycopg2

    conn = psycopg2.connect(database_url)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS wazuh_ingest_state (
                    id SERIAL PRIMARY KEY,
                    key_name TEXT NOT NULL UNIQUE,
                    last_timestamp_ms BIGINT,
                    last_pit_id TEXT,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
    finally:
        conn.close()


def insert_analysis_run(database_url: str, row: dict[str, Any]) -> int:
    import psycopg2
    from psycopg2.extras import Json

    def _j(val: Any) -> Any:
        if val is None:
            return None
        return Json(val)

    ensure_analysis_runs_schema(database_url)
    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO analysis_runs (
                    source, dataset_label, episode_id, run_id, target_ip,
                    window_start_ms, window_end_ms, alert_count, status,
                    evidence_json, agent_outputs_json, issues_json, error_message,
                    writeback_json, layerc_summary
                ) VALUES (
                    %(source)s, %(dataset_label)s, %(episode_id)s, %(run_id)s, %(target_ip)s,
                    %(window_start_ms)s, %(window_end_ms)s, %(alert_count)s, %(status)s,
                    %(evidence_json)s, %(agent_outputs_json)s, %(issues_json)s, %(error_message)s,
                    %(writeback_json)s, %(layerc_summary)s
                )
                RETURNING id
                """,
                {
                    "source": row["source"],
                    "dataset_label": row.get("dataset_label"),
                    "episode_id": row["episode_id"],
                    "run_id": row["run_id"],
                    "target_ip": row.get("target_ip"),
                    "window_start_ms": row["window_start_ms"],
                    "window_end_ms": row["window_end_ms"],
                    "alert_count": row.get("alert_count", 0),
                    "status": row["status"],
                    "evidence_json": _j(row.get("evidence_json")),
                    "agent_outputs_json": _j(row.get("agent_outputs_json")),
                    "issues_json": _j(row.get("issues_json")),
                    "error_message": row.get("error_message"),
                    "writeback_json": _j(row.get("writeback_json")),
                    "layerc_summary": _j(row.get("layerc_summary")),
                },
            )
            new_id = cur.fetchone()[0]
        conn.commit()
        return int(new_id)
    finally:
        conn.close()


def list_runs(
    database_url: str,
    *,
    limit: int = 50,
    offset: int = 0,
    source: Optional[str] = None,
) -> list[dict[str, Any]]:
    import psycopg2
    from psycopg2.extras import RealDictCursor

    ensure_analysis_runs_schema(database_url)
    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if source:
                cur.execute(
                    """
                    SELECT id, created_at, source, dataset_label, episode_id, run_id, target_ip,
                           window_start_ms, window_end_ms, alert_count, status, error_message,
                           layerc_summary
                    FROM analysis_runs WHERE source = %s
                    ORDER BY created_at DESC LIMIT %s OFFSET %s
                    """,
                    (source, limit, offset),
                )
            else:
                cur.execute(
                    """
                    SELECT id, created_at, source, dataset_label, episode_id, run_id, target_ip,
                           window_start_ms, window_end_ms, alert_count, status, error_message,
                           layerc_summary
                    FROM analysis_runs
                    ORDER BY created_at DESC LIMIT %s OFFSET %s
                    """,
                    (limit, offset),
                )
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def get_run_by_id(database_url: str, run_id: int) -> Optional[dict[str, Any]]:
    import psycopg2
    from psycopg2.extras import RealDictCursor

    ensure_analysis_runs_schema(database_url)
    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM analysis_runs WHERE id = %s", (run_id,))
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def get_adjacent_run_ids(database_url: str, run_id: int) -> tuple[Optional[int], Optional[int]]:
    """
    與 list_runs 相同排序（created_at DESC, id DESC）下的相鄰 id。
    回傳 (較新一筆, 較舊一筆)：較新 = 列表中較上方列。
    """
    import psycopg2

    ensure_analysis_runs_schema(database_url)
    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH ordered AS (
                    SELECT id,
                           LAG(id) OVER (ORDER BY created_at DESC, id DESC) AS newer_id,
                           LEAD(id) OVER (ORDER BY created_at DESC, id DESC) AS older_id
                    FROM analysis_runs
                )
                SELECT newer_id, older_id FROM ordered WHERE id = %s
                """,
                (run_id,),
            )
            row = cur.fetchone()
            if not row:
                return None, None
            n, o = row[0], row[1]
            return (int(n) if n is not None else None, int(o) if o is not None else None)
    finally:
        conn.close()


def get_run_by_episode_id(database_url: str, episode_id: str) -> Optional[dict[str, Any]]:
    import psycopg2
    from psycopg2.extras import RealDictCursor

    ensure_analysis_runs_schema(database_url)
    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM analysis_runs WHERE episode_id = %s ORDER BY created_at DESC LIMIT 1",
                (episode_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()
