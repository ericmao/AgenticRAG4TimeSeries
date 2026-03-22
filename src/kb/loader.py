"""
KB 列舉與讀取：KB_DB_MODE=off|merge|db_only（見 src.config.Config）。
merge：同 rel_path 以資料庫內容為準（列舉與 read 一致）。

透過 kb_browser 模組引用函式，以便測試 monkeypatch kb_root_abs。
"""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import services.mvp_ui_api.kb_browser as _kb_browser

_MAX_BYTES_GROUPING = 2_000_000


def get_kb_db_mode() -> str:
    from src.config import get_config

    v = (get_config().KB_DB_MODE or "off").strip().lower()
    if v in ("db_only", "db-only"):
        return "db_only"
    if v == "merge":
        return "merge"
    return "off"


def _database_url() -> Optional[str]:
    u = os.environ.get("DATABASE_URL", "").strip()
    return u or None


def _row_from_full_text(rel: str, full: str, size: int, mtime: int) -> dict[str, Any]:
    seq, pat = _kb_browser.parse_kb_sequence_and_pattern(full)
    gk = _kb_browser.group_key_from_text(full)
    return {
        "rel_path": rel,
        "size": size,
        "mtime": mtime,
        "sequence_tags": seq or "",
        "pattern_summary": pat or "",
        "group_key": gk,
        "kb_source": "filesystem",
    }


def _scan_filesystem(repo_root: Path) -> list[dict[str, Any]]:
    root = _kb_browser.kb_root_abs(repo_root)
    if not root.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".md", ".txt"):
            continue
        rel = p.relative_to(root).as_posix()
        if _kb_browser._is_kb_index_excluded(rel):
            continue
        st = p.stat()
        full = _kb_browser._read_text_capped(p, _MAX_BYTES_GROUPING)
        rows.append(
            _row_from_full_text(rel, full, st.st_size, int(st.st_mtime))
        )
    return rows


def _row_from_db_record(rec: dict[str, Any]) -> dict[str, Any]:
    body = rec["body"]
    rel = rec["rel_path"]
    size = len(body.encode("utf-8", errors="replace"))
    updated_at = rec.get("updated_at")
    if updated_at is not None and hasattr(updated_at, "timestamp"):
        mtime = int(updated_at.timestamp())
    else:
        mtime = 0
    seq, pat = _kb_browser.parse_kb_sequence_and_pattern(body)
    gk = _kb_browser.group_key_from_text(body)
    return {
        "rel_path": rel,
        "size": size,
        "mtime": mtime,
        "sequence_tags": seq or "",
        "pattern_summary": pat or "",
        "group_key": gk,
        "kb_source": "database",
        "db_version": rec.get("version"),
    }


def list_document_rows(repo_root: Path) -> list[dict[str, Any]]:
    mode = get_kb_db_mode()
    fs_rows = _scan_filesystem(repo_root) if mode in ("off", "merge") else []

    if mode == "off":
        rows = fs_rows
    elif mode == "db_only":
        db_url = _database_url()
        if not db_url:
            return []
        from src.storage.kb_documents_store import list_all_kb_documents

        db_recs = list_all_kb_documents(db_url)
        rows = [_row_from_db_record(r) for r in db_recs]
    else:
        # merge
        db_url = _database_url()
        by_path: dict[str, dict[str, Any]] = {r["rel_path"]: r for r in fs_rows}
        if db_url:
            from src.storage.kb_documents_store import list_all_kb_documents

            for rec in list_all_kb_documents(db_url):
                by_path[rec["rel_path"]] = _row_from_db_record(rec)
        rows = list(by_path.values())

    rows.sort(key=lambda x: (x["group_key"], x["rel_path"]))
    counts = Counter(r["group_key"] for r in rows)
    for r in rows:
        r["group_size"] = counts[r["group_key"]]
    return rows


def read_kb_body(repo_root: Path, rel_path: str) -> str:
    """讀取 KB 全文（merge：DB 優先；db_only：僅 DB；off：僅檔案）。"""
    mode = get_kb_db_mode()
    db_url = _database_url()

    if mode == "db_only":
        if not db_url:
            raise FileNotFoundError(rel_path)
        from src.storage.kb_documents_store import get_kb_document_by_path

        rec = get_kb_document_by_path(db_url, rel_path)
        if not rec:
            raise FileNotFoundError(rel_path)
        return rec["body"]

    if mode == "merge" and db_url:
        from src.storage.kb_documents_store import get_kb_document_by_path

        rec = get_kb_document_by_path(db_url, rel_path)
        if rec:
            return rec["body"]

    root = _kb_browser.kb_root_abs(repo_root)
    path = root / rel_path
    if not path.is_file():
        raise FileNotFoundError(rel_path)
    return _kb_browser._read_text_capped(path, _MAX_BYTES_GROUPING)


def get_db_version_if_any(rel_path: str) -> Optional[int]:
    """若該路徑在 DB 有列則回傳 version，否則 None。"""
    if get_kb_db_mode() == "off":
        return None
    db_url = _database_url()
    if not db_url:
        return None
    from src.storage.kb_documents_store import get_kb_document_by_path

    rec = get_kb_document_by_path(db_url, rel_path)
    return int(rec["version"]) if rec else None
