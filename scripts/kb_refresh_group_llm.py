#!/usr/bin/env python3
"""
批次為每個 KB 模板群組呼叫 LLM，寫入 outputs/.kb_group_llm_cache.json；
  若設定 DATABASE_URL，同步寫入 PostgreSQL 表 kb_group_llm。

前置：Ollama（或依 src/llm/factory.py 設定之 LLM_BACKEND）。

用法:
  cd AgenticRAG4TimeSeries && PYTHONPATH=. python scripts/kb_refresh_group_llm.py
  PYTHONPATH=. python scripts/kb_refresh_group_llm.py --max-groups 3 --dry-run
  PYTHONPATH=. python scripts/kb_refresh_group_llm.py --force
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
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

from langchain_core.messages import HumanMessage

from services.mvp_ui_api.kb_browser import _read_head, aggregate_kb_groups, kb_root_abs
from services.mvp_ui_api.kb_group_llm import load_llm_cache, save_llm_cache
from src.kb.group_llm_prompt import build_group_llm_prompt, parse_llm_group_json
from src.llm.factory import get_llm_for_layer_c


def _fingerprint_for_group(repo_root: Path, group: dict) -> str:
    kb = kb_root_abs(repo_root)
    chunks: list[str] = []
    for f in sorted(group["files"], key=lambda x: x["rel_path"]):
        rel = f["rel_path"]
        p = kb / rel
        if not p.is_file():
            continue
        body = _read_head(p, max_chars=12_000)
        h = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
        chunks.append(f"{rel}:{h}")
    return hashlib.sha256("\n".join(chunks).encode("utf-8")).hexdigest()[:32]


def _collect_excerpts(repo_root: Path, group: dict, *, per_file: int, total_cap: int) -> str:
    kb = kb_root_abs(repo_root)
    parts: list[str] = []
    n = 0
    for f in sorted(group["files"], key=lambda x: x["rel_path"]):
        rel = f["rel_path"]
        p = kb / rel
        if not p.is_file():
            continue
        chunk = _read_head(p, max_chars=per_file)
        parts.append(f"### 檔案: {rel}\n{chunk}")
        n += len(chunk)
        if n >= total_cap:
            break
    text = "\n\n".join(parts)
    if len(text) > total_cap:
        text = text[:total_cap].rstrip() + "\n…（摘錄已截斷）"
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description="Refresh KB group LLM cache (meaning / usage / threats).")
    ap.add_argument("--max-groups", type=int, default=0, help="最多處理群組數（0 表示全部）")
    ap.add_argument("--per-file-chars", type=int, default=6000, help="每檔摘錄字元上限")
    ap.add_argument("--total-chars", type=int, default=24_000, help="群組摘錄總字元上限")
    ap.add_argument("--force", action="store_true", help="忽略 fingerprint，全部重跑")
    ap.add_argument("--dry-run", action="store_true", help="只列出將處理的 group_key，不呼叫 LLM")
    args = ap.parse_args()

    groups = aggregate_kb_groups(REPO_ROOT)
    cache = load_llm_cache(REPO_ROOT)
    llm = None
    model_name = ""
    if not args.dry_run:
        llm = get_llm_for_layer_c()
        try:
            from src.config import get_config

            cfg = get_config()
            model_name = (cfg.LLM_MODEL or "").strip() or "unknown"
        except Exception:
            model_name = "unknown"

    processed = 0
    for g in groups:
        if args.max_groups and processed >= args.max_groups:
            break
        gk = g["group_key"]
        fp = _fingerprint_for_group(REPO_ROOT, g)
        prev = cache.get(gk) if isinstance(cache.get(gk), dict) else None
        if not args.force and isinstance(prev, dict) and prev.get("content_fingerprint") == fp:
            print(f"skip (unchanged) {gk}")
            continue
        if args.dry_run:
            print(f"would refresh {gk} fp={fp}")
            processed += 1
            continue

        excerpts = _collect_excerpts(
            REPO_ROOT,
            g,
            per_file=args.per_file_chars,
            total_cap=args.total_chars,
        )
        prompt = build_group_llm_prompt(
            group_key=gk,
            group_size=g["group_size"],
            representative_rel_path=g.get("representative_rel_path") or "",
            excerpts=excerpts or "（無摘錄）",
        )
        assert llm is not None
        resp = llm.invoke([HumanMessage(content=prompt)])
        raw = getattr(resp, "content", None) or str(resp)
        parsed = parse_llm_group_json(raw)
        now = datetime.now(timezone.utc).isoformat()
        ent = {
            "meaning": parsed["meaning"],
            "usage_direction": parsed["usage_direction"],
            "threats": parsed["threats"],
            "content_fingerprint": fp,
            "updated_at": now,
            "model": model_name,
        }
        # 保留舊版 summary 若曾手動維護且本次未再產生（可選：清空）
        if isinstance(prev, dict) and (prev.get("summary") or "").strip():
            ent["summary"] = prev["summary"].strip()
        cache[gk] = ent
        save_llm_cache(REPO_ROOT, cache)
        db_url = (os.environ.get("DATABASE_URL") or "").strip()
        if db_url:
            try:
                from src.storage.kb_group_llm_store import upsert_kb_group_llm

                upsert_kb_group_llm(
                    db_url,
                    group_key=gk,
                    meaning=parsed["meaning"],
                    usage_direction=parsed["usage_direction"],
                    threats=parsed["threats"],
                    summary=(ent.get("summary") or "").strip(),
                    content_fingerprint=fp,
                    model=model_name,
                )
            except Exception as e:
                print(f"warn: DB upsert failed for {gk}: {e}")
        print(f"ok {gk} meaning_len={len(parsed['meaning'])}")
        processed += 1

    if args.dry_run:
        print(f"dry-run: {processed} group(s) would be refreshed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
