"""
每個 KB 檔案一句繁中概要：欄位 llm_file_summary 僅從 outputs/.kb_file_llm_cache.json 套用。
瀏覽 /kb 時不呼叫 LLM；若需寫入快取請用批次腳本或離線流程。
鍵為 rel_path；content_sha256 須與目前檔案內容一致才顯示摘要。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_CACHE_FILENAME = ".kb_file_llm_cache.json"


def _cache_path(repo_root: Path) -> Path:
    out = repo_root / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out / _CACHE_FILENAME


def load_file_llm_cache(repo_root: Path) -> dict[str, Any]:
    p = _cache_path(repo_root)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_file_llm_cache(repo_root: Path, cache: dict[str, Any]) -> None:
    _cache_path(repo_root).write_text(
        json.dumps(cache, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _sha256_short(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:32]


def enrich_files_llm_summaries(repo_root: Path, groups: list[dict[str, Any]]) -> None:
    """
    就地為每個 file 列加上 llm_file_summary：僅從快取讀取，不呼叫 LLM。
    """
    from src.kb.loader import read_kb_body

    cache = load_file_llm_cache(repo_root)
    for g in groups:
        for f in g.get("files") or []:
            rel = f.get("rel_path")
            if not rel:
                f.setdefault("llm_file_summary", "")
                continue
            try:
                raw = read_kb_body(repo_root, rel)
            except Exception:
                f["llm_file_summary"] = ""
                continue
            sha = _sha256_short(raw)
            entry = cache.get(rel) if isinstance(cache.get(rel), dict) else {}
            if entry.get("content_sha256") == sha and (entry.get("summary") or "").strip():
                f["llm_file_summary"] = entry["summary"].strip()
            else:
                f["llm_file_summary"] = ""
