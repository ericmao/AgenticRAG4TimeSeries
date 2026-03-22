#!/usr/bin/env python3
"""
持續監看 KB：對新檔或內容變更呼叫 Ollama 產生單檔 description（summary），
寫入 outputs/.kb_file_llm_cache.json（與 /kb「LLM 說明」同源）。

設計對齊 OpenClaw「長駐 worker」：本腳本為輪詢迴圈；單次任務可改用
  src.layer_c.agents.kb_describer_agent.register_kb_describer + agent_id kb_describer。

環境：LLM_OLLAMA_PRIMARY、LLM_MODEL、KB_PATH（可選）

用法:
  cd AgenticRAG4TimeSeries && PYTHONPATH=. python scripts/kb_watch_llm.py --once
  PYTHONPATH=. python scripts/kb_watch_llm.py --interval-sec 300
  PYTHONPATH=. python scripts/kb_watch_llm.py --interval-sec 60 --max-files-per-cycle 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from src.kb.file_llm_refresh import result_to_dict, run_kb_file_llm_refresh


def _write_activity(repo_root: Path, payload: dict) -> None:
    out = repo_root / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    p = out / "kb_watch_activity.json"
    p.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Poll KB and refresh LLM file summaries when needed.")
    ap.add_argument("--interval-sec", type=int, default=300, help="輪詢間隔秒數（--once 時忽略）")
    ap.add_argument("--once", action="store_true", help="只跑一輪後退出")
    ap.add_argument("--force", action="store_true", help="每輪皆 --force（慎用，成本高）")
    ap.add_argument(
        "--max-files-per-cycle",
        type=int,
        default=0,
        help="每輪最多處理檔案數（0=本輪需更新的全跑；可避免長時間阻塞）",
    )
    ap.add_argument("--max-chars", type=int, default=48_000, help="送入 LLM 的每檔最大字元")
    ap.add_argument(
        "--no-activity-file",
        action="store_true",
        help="不寫 outputs/kb_watch_activity.json",
    )
    args = ap.parse_args()

    cycle = 0
    while True:
        cycle += 1
        t0 = time.monotonic()
        r = run_kb_file_llm_refresh(
            REPO_ROOT,
            force=args.force,
            max_files=args.max_files_per_cycle or 0,
            max_chars=args.max_chars,
            dry_run=False,
        )
        elapsed = time.monotonic() - t0
        now = datetime.now(timezone.utc).isoformat()
        record = {
            "schema": "kb_watch.v1",
            "updated_at": now,
            "cycle": cycle,
            "elapsed_sec": round(elapsed, 3),
            "stats": result_to_dict(r),
        }
        if not args.no_activity_file:
            _write_activity(REPO_ROOT, record)
        print(
            f"[kb_watch] cycle={cycle} refreshed={r.refreshed} "
            f"skipped_unchanged={r.skipped_unchanged} errors={len(r.errors)} "
            f"elapsed={elapsed:.1f}s"
        )
        for rel, ln in r.refreshed_details:
            print(f"  ok {rel} len={ln}")
        for e in r.errors:
            print(f"  err {e}")
        if args.once:
            return 0 if not r.errors else 1
        time.sleep(max(1, args.interval_sec))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[kb_watch] stopped", file=sys.stderr)
        raise SystemExit(130)
