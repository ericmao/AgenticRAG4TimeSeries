"""
單檔 KB → LLM 概要：寫入 outputs/.kb_file_llm_cache.json。
供 scripts/kb_refresh_file_llm.py、kb_watch_llm.py、OpenClaw kb_describer 共用。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from services.mvp_ui_api.kb_browser import list_kb_documents
from services.mvp_ui_api.kb_file_llm import load_file_llm_cache, save_file_llm_cache
from src.kb.file_llm_prompt import build_file_llm_prompt, parse_file_summary_json
from src.kb.loader import read_kb_body
from src.llm.factory import get_llm_for_layer_c


def _sha256_short(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:32]


@dataclass
class KbFileLlmRefreshResult:
    refreshed: int = 0
    skipped_unchanged: int = 0
    skipped_read_error: int = 0
    dry_run_would: int = 0
    paths_refreshed: list[str] = field(default_factory=list)
    refreshed_details: list[tuple[str, int]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def run_kb_file_llm_refresh(
    repo_root: Path,
    *,
    force: bool = False,
    max_files: int = 0,
    max_chars: int = 48_000,
    dry_run: bool = False,
) -> KbFileLlmRefreshResult:
    """
    掃描 KB 目錄；對「新檔／內容變更／缺 summary」的檔案呼叫 LLM，寫入快取。
    未變且已有 summary 則跳過（除非 force）。
    """
    out = KbFileLlmRefreshResult()
    rows = list_kb_documents(repo_root)
    cache = load_file_llm_cache(repo_root)
    llm = None
    model_name = ""
    if not dry_run:
        llm = get_llm_for_layer_c()
        try:
            from src.config import get_config

            model_name = (get_config().LLM_MODEL or "").strip() or "unknown"
        except Exception:
            model_name = "unknown"

    processed = 0
    for row in sorted(rows, key=lambda r: r.get("rel_path") or ""):
        if max_files and processed >= max_files:
            break
        rel = row.get("rel_path")
        if not rel:
            continue
        try:
            raw = read_kb_body(repo_root, rel)
        except Exception as e:
            out.skipped_read_error += 1
            out.errors.append(f"{rel}: {e}")
            continue
        sha = _sha256_short(raw)
        prev = cache.get(rel) if isinstance(cache.get(rel), dict) else None
        if (
            not force
            and isinstance(prev, dict)
            and prev.get("content_sha256") == sha
            and (prev.get("summary") or "").strip()
        ):
            out.skipped_unchanged += 1
            continue
        excerpt = raw if len(raw) <= max_chars else raw[:max_chars].rstrip() + "\n…（摘錄已截斷）"
        if dry_run:
            out.dry_run_would += 1
            out.paths_refreshed.append(rel)
            processed += 1
            continue

        prompt = build_file_llm_prompt(rel_path=rel, excerpt=excerpt or "（空檔）")
        assert llm is not None
        try:
            resp = llm.invoke([HumanMessage(content=prompt)])
        except Exception as e:
            out.errors.append(f"{rel} invoke: {e}")
            continue
        text = getattr(resp, "content", None) or str(resp)
        summary = parse_file_summary_json(text)
        if not summary:
            out.errors.append(f"{rel}: empty summary, raw head={text[:120]!r}")
        now = datetime.now(timezone.utc).isoformat()
        cache[rel] = {
            "summary": summary,
            "content_sha256": sha,
            "updated_at": now,
            "model": model_name,
        }
        save_file_llm_cache(repo_root, cache)
        out.refreshed += 1
        out.paths_refreshed.append(rel)
        out.refreshed_details.append((rel, len(summary)))
        processed += 1

    return out


def result_to_dict(r: KbFileLlmRefreshResult) -> dict[str, Any]:
    return {
        "refreshed": r.refreshed,
        "skipped_unchanged": r.skipped_unchanged,
        "skipped_read_error": r.skipped_read_error,
        "dry_run_would": r.dry_run_would,
        "paths_refreshed": r.paths_refreshed,
        "refreshed_details": [{"path": a, "summary_len": b} for a, b in r.refreshed_details],
        "errors": r.errors,
    }
