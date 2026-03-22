"""
KB 群組繁中摘要：欄位 llm_summary 僅從 outputs/.kb_group_llm_cache.json 讀取。
瀏覽 /kb 時不呼叫 LLM；若需寫入快取請用批次腳本或離線流程。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


def enrich_groups_llm_summaries(repo_root: Path, groups: list[dict[str, Any]]) -> None:
    """
    就地為每個 group 加上 llm_summary：僅從快取讀取，不呼叫 LLM。
    """
    cache = load_llm_cache(repo_root)
    for g in groups:
        gk = g["group_key"]
        ent = cache.get(gk) if isinstance(cache.get(gk), dict) else None
        if ent and (ent.get("summary") or "").strip():
            g["llm_summary"] = ent["summary"].strip()
        else:
            g["llm_summary"] = ""
