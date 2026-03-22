#!/usr/bin/env python3
"""
依序執行單檔與群組 KB LLM 快取更新（outputs/.kb_file_llm_cache.json、.kb_group_llm_cache.json）。

用法:
  cd AgenticRAG4TimeSeries && PYTHONPATH=. python scripts/kb_refresh_all_llm.py
  PYTHONPATH=. python scripts/kb_refresh_all_llm.py --force
  PYTHONPATH=. python scripts/kb_refresh_all_llm.py --dry-run --max-files 3 --max-groups 2
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Run kb_refresh_file_llm.py then kb_refresh_group_llm.py")
    ap.add_argument("--force", action="store_true", help="兩支腳本皆 --force")
    ap.add_argument("--dry-run", action="store_true", help="兩支腳本皆 --dry-run")
    ap.add_argument("--max-files", type=int, default=0, help="僅傳給 kb_refresh_file_llm.py（0=全部）")
    ap.add_argument("--max-chars", type=int, default=0, help="僅傳給 kb_refresh_file_llm.py（0=用該腳本預設）")
    ap.add_argument("--max-groups", type=int, default=0, help="僅傳給 kb_refresh_group_llm.py（0=全部）")
    ap.add_argument("--per-file-chars", type=int, default=0, help="僅傳給 kb_refresh_group_llm.py")
    ap.add_argument("--total-chars", type=int, default=0, help="僅傳給 kb_refresh_group_llm.py")
    args = ap.parse_args()

    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}

    file_cmd = [sys.executable, str(REPO_ROOT / "scripts" / "kb_refresh_file_llm.py")]
    if args.force:
        file_cmd.append("--force")
    if args.dry_run:
        file_cmd.append("--dry-run")
    if args.max_files:
        file_cmd += ["--max-files", str(args.max_files)]
    if args.max_chars:
        file_cmd += ["--max-chars", str(args.max_chars)]

    group_cmd = [sys.executable, str(REPO_ROOT / "scripts" / "kb_refresh_group_llm.py")]
    if args.force:
        group_cmd.append("--force")
    if args.dry_run:
        group_cmd.append("--dry-run")
    if args.max_groups:
        group_cmd += ["--max-groups", str(args.max_groups)]
    if args.per_file_chars:
        group_cmd += ["--per-file-chars", str(args.per_file_chars)]
    if args.total_chars:
        group_cmd += ["--total-chars", str(args.total_chars)]

    for cmd in (file_cmd, group_cmd):
        print(f"+ {' '.join(cmd)}")
        r = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
        if r.returncode != 0:
            return r.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
