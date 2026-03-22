"""PostgreSQL：kb_documents / kb_document_versions 建立、讀寫、樂觀鎖更新。"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

_SCHEMA_ENSURED = False


def ensure_kb_documents_schema(database_url: str) -> None:
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
                CREATE TABLE IF NOT EXISTS kb_documents (
                    id SERIAL PRIMARY KEY,
                    rel_path TEXT NOT NULL UNIQUE,
                    body TEXT NOT NULL,
                    version INT NOT NULL DEFAULT 1,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_by TEXT
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS kb_document_versions (
                    id SERIAL PRIMARY KEY,
                    document_id INT NOT NULL REFERENCES kb_documents(id) ON DELETE CASCADE,
                    version INT NOT NULL,
                    body TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    editor TEXT,
                    note TEXT,
                    UNIQUE (document_id, version)
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_kb_document_versions_doc_ver
                ON kb_document_versions(document_id, version DESC)
                """
            )
        _SCHEMA_ENSURED = True
    finally:
        conn.close()


def reset_schema_cache_for_tests() -> None:
    global _SCHEMA_ENSURED
    _SCHEMA_ENSURED = False


def list_all_kb_documents(database_url: str) -> list[dict[str, Any]]:
    ensure_kb_documents_schema(database_url)
    import psycopg2

    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT rel_path, body, version, updated_at, updated_by, id
                FROM kb_documents
                ORDER BY rel_path
                """
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    out: list[dict[str, Any]] = []
    for rel_path, body, version, updated_at, updated_by, doc_id in rows:
        out.append(
            {
                "id": doc_id,
                "rel_path": rel_path,
                "body": body,
                "version": version,
                "updated_at": updated_at,
                "updated_by": updated_by,
            }
        )
    return out


def get_kb_document_by_path(database_url: str, rel_path: str) -> Optional[dict[str, Any]]:
    ensure_kb_documents_schema(database_url)
    import psycopg2

    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, rel_path, body, version, updated_at, updated_by
                FROM kb_documents
                WHERE rel_path = %s
                """,
                (rel_path,),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    if not row:
        return None
    doc_id, rel_path, body, version, updated_at, updated_by = row
    return {
        "id": doc_id,
        "rel_path": rel_path,
        "body": body,
        "version": version,
        "updated_at": updated_at,
        "updated_by": updated_by,
    }


def list_kb_document_versions(
    database_url: str, rel_path: str, limit: int = 50
) -> list[dict[str, Any]]:
    ensure_kb_documents_schema(database_url)
    import psycopg2

    lim = max(1, min(limit, 200))
    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT v.version, v.body, v.created_at, v.editor, v.note
                FROM kb_document_versions v
                JOIN kb_documents d ON d.id = v.document_id
                WHERE d.rel_path = %s
                ORDER BY v.version DESC
                LIMIT %s
                """,
                (rel_path, lim),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    out: list[dict[str, Any]] = []
    for version, body, created_at, editor, note in rows:
        out.append(
            {
                "version": version,
                "body": body,
                "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at),
                "editor": editor,
                "note": note,
            }
        )
    return out


class KbVersionConflict(Exception):
    def __init__(self, current_version: int):
        self.current_version = current_version
        super().__init__(f"version conflict: expected different from {current_version}")


def upsert_kb_document(
    database_url: str,
    rel_path: str,
    body: str,
    expected_version: int,
    editor: Optional[str] = None,
    note: Optional[str] = None,
) -> dict[str, Any]:
    """
    新建：expected_version 必須為 0，且 rel_path 尚不存在。
    更新：expected_version 必須等於目前 kb_documents.version。
    衝突時 raise KbVersionConflict(current_version)。
    """
    ensure_kb_documents_schema(database_url)
    import psycopg2

    conn = psycopg2.connect(database_url)
    try:
        conn.autocommit = False
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, version FROM kb_documents WHERE rel_path = %s FOR UPDATE",
                (rel_path,),
            )
            row = cur.fetchone()
            now = datetime.now(timezone.utc)

            if row is None:
                if expected_version != 0:
                    conn.rollback()
                    raise KbVersionConflict(0)
                cur.execute(
                    """
                    INSERT INTO kb_documents (rel_path, body, version, updated_at, updated_by)
                    VALUES (%s, %s, 1, %s, %s)
                    RETURNING id, version, updated_at
                    """,
                    (rel_path, body, now, editor),
                )
                doc_id, new_ver, updated_at = cur.fetchone()
                cur.execute(
                    """
                    INSERT INTO kb_document_versions (document_id, version, body, editor, note)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (doc_id, 1, body, editor, note),
                )
                conn.commit()
                return {
                    "rel_path": rel_path,
                    "version": new_ver,
                    "updated_at": updated_at.isoformat() if hasattr(updated_at, "isoformat") else str(updated_at),
                }

            doc_id, current_ver = row
            if expected_version != current_ver:
                conn.rollback()
                raise KbVersionConflict(current_ver)

            new_ver = current_ver + 1
            cur.execute(
                """
                UPDATE kb_documents
                SET body = %s, version = %s, updated_at = %s, updated_by = %s
                WHERE id = %s AND version = %s
                RETURNING updated_at
                """,
                (body, new_ver, now, editor, doc_id, current_ver),
            )
            upd = cur.fetchone()
            if not upd:
                conn.rollback()
                with conn.cursor() as cur2:
                    cur2.execute("SELECT version FROM kb_documents WHERE rel_path = %s", (rel_path,))
                    vrow = cur2.fetchone()
                raise KbVersionConflict(int(vrow[0]) if vrow else current_ver)
            updated_at = upd[0]
            cur.execute(
                """
                INSERT INTO kb_document_versions (document_id, version, body, editor, note)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (doc_id, new_ver, body, editor, note),
            )
            conn.commit()
            return {
                "rel_path": rel_path,
                "version": new_ver,
                "updated_at": updated_at.isoformat() if hasattr(updated_at, "isoformat") else str(updated_at),
            }
    finally:
        conn.close()
