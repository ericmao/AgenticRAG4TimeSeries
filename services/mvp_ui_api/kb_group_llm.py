"""
KB 群組繁中摘要：優先讀 PostgreSQL kb_group_llm；否則 outputs/.kb_group_llm_cache.json。
支援 legacy「summary」與 meaning / usage_direction / threats（批次腳本寫入）。
瀏覽 /kb 時不呼叫 LLM；批次見 scripts/kb_refresh_group_llm.py。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

_CACHE_FILENAME = ".kb_group_llm_cache.json"


def _cache_path(repo_root: Path) -> Path:
    out = repo_root / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out / _CACHE_FILENAME


def load_llm_cache(repo_root: Path) -> dict[str, Any]:
    p = _cache_path(repo_root)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_llm_cache(repo_root: Path, cache: dict[str, Any]) -> None:
    _cache_path(repo_root).write_text(
        json.dumps(cache, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _apply_ent_to_group(g: dict[str, Any], ent: Optional[dict[str, Any]]) -> None:
    g["llm_meaning"] = ""
    g["llm_usage_direction"] = ""
    g["llm_threats"] = ""
    g["llm_summary"] = ""
    if not ent:
        return
    meaning = (ent.get("meaning") or "").strip()
    usage = (ent.get("usage_direction") or "").strip()
    threats = (ent.get("threats") or "").strip()
    legacy = (ent.get("summary") or "").strip()
    g["llm_meaning"] = meaning
    g["llm_usage_direction"] = usage
    g["llm_threats"] = threats
    if legacy:
        g["llm_summary"] = legacy
    elif meaning:
        g["llm_summary"] = meaning[:240] + ("…" if len(meaning) > 240 else "")
    else:
        g["llm_summary"] = ""


def enrich_groups_llm_summaries(repo_root: Path, groups: list[dict[str, Any]]) -> None:
    """
    就地為每個 group 加上 llm_summary、llm_meaning、llm_usage_direction、llm_threats。
    若 DATABASE_URL 且 kb_group_llm 有該 group_key，以資料庫為準；否則用 JSON 快取。
    """
    cache = load_llm_cache(repo_root)
    db_by_gk: dict[str, dict[str, Any]] = {}
    db_url = (os.environ.get("DATABASE_URL") or "").strip()
    if db_url:
        try:
            from src.storage.kb_group_llm_store import fetch_all_kb_group_llm

            db_by_gk = fetch_all_kb_group_llm(db_url)
        except Exception:
            db_by_gk = {}

    for g in groups:
        gk = g["group_key"]
        if gk in db_by_gk:
            ent = db_by_gk[gk]
        else:
            ent = cache.get(gk) if isinstance(cache.get(gk), dict) else None
        _apply_ent_to_group(g, ent)
