#!/usr/bin/env python3
"""
批次為每個 KB 檔案呼叫 LLM，寫入 outputs/.kb_file_llm_cache.json：
  summary（繁中概要）、content_sha256（與 read_kb_body 一致時 /kb 才顯示）。

前置：Ollama 可連（見 LLM_OLLAMA_PRIMARY、LLM_MODEL）。

用法:
  cd AgenticRAG4TimeSeries && PYTHONPATH=. python scripts/kb_refresh_file_llm.py
  PYTHONPATH=. python scripts/kb_refresh_file_llm.py --max-files 5 --dry-run
  PYTHONPATH=. python scripts/kb_refresh_file_llm.py --force

持續監看新檔／變更：見 scripts/kb_watch_llm.py；OpenClaw 單次任務：agent_id kb_describer。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from src.kb.file_llm_refresh import run_kb_file_llm_refresh


def main() -> int:
    ap = argparse.ArgumentParser(description="Refresh per-file KB LLM cache (summary).")
    ap.add_argument("--max-files", type=int, default=0, help="最多處理檔案數（0 表示全部）")
    ap.add_argument("--max-chars", type=int, default=48_000, help="每檔送入 LLM 的最大字元數")
    ap.add_argument("--force", action="store_true", help="略過「內容未變」跳過邏輯，全部重跑")
    ap.add_argument("--dry-run", action="store_true", help="只列出將處理的 rel_path")
    args = ap.parse_args()

    r = run_kb_file_llm_refresh(
        REPO_ROOT,
        force=args.force,
        max_files=args.max_files,
        max_chars=args.max_chars,
        dry_run=args.dry_run,
    )
    for err in r.errors:
        print(f"warn/error: {err}")
    if args.dry_run:
        for p in r.paths_refreshed:
            print(f"would refresh {p}")
        print(f"dry-run: {r.dry_run_would} file(s)")
    else:
        for rel, ln in r.refreshed_details:
            print(f"ok {rel} len={ln}")
        if r.refreshed == 0 and r.skipped_unchanged:
            print(
                f"skip (unchanged) x{r.skipped_unchanged} file(s) — use --force to re-run LLM"
            )
        elif r.skipped_unchanged:
            print(
                f"note: skipped_unchanged={r.skipped_unchanged} (content+summary already in cache)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
