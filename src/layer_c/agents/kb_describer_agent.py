"""
OpenClaw 風格：單次任務為「掃描 KB → 為新／變更檔產生 LLM 概要（description）」。
持續輪詢請用 scripts/kb_watch_llm.py。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.kb.file_llm_refresh import result_to_dict, run_kb_file_llm_refresh
from src.layer_c.runtime.openclaw_adapter import OpenClawAdapter


def kb_describer_runner(payload: dict[str, Any]) -> dict[str, Any]:
    """
    payload:
      repo_root: str（可選，預設 cwd 上兩層的 repo 根 — 呼叫端應傳 REPO_ROOT）
      force: bool
      max_files: int（0=全部）
      max_chars: int
      dry_run: bool
    """
    raw_root = payload.get("repo_root")
    if raw_root:
        repo_root = Path(str(raw_root)).resolve()
    else:
        repo_root = Path(__file__).resolve().parents[3]
    force = bool(payload.get("force"))
    max_files = int(payload.get("max_files") or 0)
    max_chars = int(payload.get("max_chars") or 48_000)
    dry_run = bool(payload.get("dry_run"))
    r = run_kb_file_llm_refresh(
        repo_root,
        force=force,
        max_files=max_files,
        max_chars=max_chars,
        dry_run=dry_run,
    )
    return {"ok": True, "stats": result_to_dict(r)}


def register_kb_describer(adapter: OpenClawAdapter) -> None:
    adapter.register("kb_describer", kb_describer_runner)
