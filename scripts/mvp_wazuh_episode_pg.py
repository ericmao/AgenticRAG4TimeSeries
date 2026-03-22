#!/usr/bin/env python3
"""
MVP: Wazuh Indexer → Episode → Layer C 完整管線 → PostgreSQL

Layer C 階段（與 src/cli retrieve / analyze / writeback 對齊）:
  C1 retrieve  — build_evidence_set → outputs/evidence/<episode_id>.json
  C2 analyze   — run_analyze_with_validation → outputs/agents/<episode_id>_*.json
  C3 writeback — run_writeback(mode) → outputs/writeback/<episode_id>.json + decision_bundle

環境變數（建議 .env）:
  WAZUH_INDEXER_URL / WAZUH_INDEXER_USERNAME / WAZUH_INDEXER_PASSWORD
  WAZUH_VERIFY_SSL=0
  DATABASE_URL
  CONTROL_PLANE_BASE_URL（API 根網址，例如 https://cti.ericmao.dev；勿含 /dashboard 或 #fragment）
  CONTROL_PLANE_TOKEN（與 BASE_URL 一併設定時才會 POST writeback）

用法:
  PYTHONPATH=. python scripts/mvp_wazuh_episode_pg.py --target-ip 192.168.1.203 --match-all --writeback-mode dry_run
  PYTHONPATH=. python scripts/mvp_wazuh_episode_pg.py --target-ip 10.0.0.5 --dry-run
  PYTHONPATH=. python scripts/mvp_wazuh_episode_pg.py ... --no-writeback   # 僅 C1+C2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from src.integrations.wazuh_indexer_client import (
    build_episode_from_events,
    build_ip_query,
    hit_to_event,
    indexer_post,
)
from src.pipeline.layer_c_run import run_layer_c_pipeline
from src.storage.analysis_runs_store import insert_analysis_run


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def main() -> int:
    parser = argparse.ArgumentParser(description="Wazuh Indexer → Episode → Layer C → PostgreSQL")
    parser.add_argument("--target-ip", required=True, help="內網 IP，例如 10.0.0.5")
    parser.add_argument("--hours", type=int, default=24, help="時間窗（小時），對應 now-Nh")
    parser.add_argument("--size", type=int, default=500, help="最多拉取 alerts 筆數")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只寫 episode JSON，不跑 retrieve / analyze / writeback、不寫 DB",
    )
    parser.add_argument("--skip-db", action="store_true", help="跑分析但不寫 PostgreSQL")
    parser.add_argument(
        "--match-all",
        action="store_true",
        help="僅依時間窗取告警，不篩 IP（Indexer 內無該 IP 或僅想試跑管線時用）",
    )
    parser.add_argument(
        "--no-writeback",
        action="store_true",
        help="略過 Layer C3 writeback（僅 C1 retrieve + C2 analyze）",
    )
    parser.add_argument(
        "--writeback-mode",
        default="dry_run",
        help="writeback 模式：dry_run（預設）、review、auto（需 OpenCTI）",
    )
    parser.add_argument(
        "--triage-rules",
        default=None,
        help="逗號分隔 rule_id；未指定則用 TRIAGE_RULES 或 episode.sequence_tags",
    )
    args = parser.parse_args()

    triage_rules = None
    if args.triage_rules and str(args.triage_rules).strip():
        triage_rules = [x.strip() for x in str(args.triage_rules).split(",") if x.strip()]

    base = os.environ.get("WAZUH_INDEXER_URL", "").strip()
    user = os.environ.get("WAZUH_INDEXER_USERNAME", "").strip()
    password = os.environ.get("WAZUH_INDEXER_PASSWORD", "")
    verify_ssl = _bool_env("WAZUH_VERIFY_SSL", default=False)
    database_url = os.environ.get("DATABASE_URL", "").strip()

    if not base or not user:
        print("請設定 WAZUH_INDEXER_URL、WAZUH_INDEXER_USERNAME、WAZUH_INDEXER_PASSWORD", file=sys.stderr)
        return 1

    run_id = f"mvp-{uuid.uuid4().hex[:16]}"
    query_body = build_ip_query(args.target_ip, args.hours, args.size, match_all=args.match_all)

    try:
        raw = indexer_post(
            base,
            "/wazuh-alerts-*/_search",
            query_body,
            user,
            password,
            verify_ssl,
        )
    except HTTPError as e:
        print(f"Indexer HTTP error: {e.code} {e.reason}", file=sys.stderr)
        try:
            print(e.read().decode()[:2000], file=sys.stderr)
        except Exception:
            pass
        return 1
    except URLError as e:
        print(f"Indexer connection error: {e}", file=sys.stderr)
        return 1

    hits = (raw.get("hits") or {}).get("hits") or []
    events = [hit_to_event(h) for h in hits]
    if not events:
        print("查無 alerts（請確認 IP 欄位、時間窗或 Wazuh 索引名稱）", file=sys.stderr)
        return 1

    try:
        episode, episode_id = build_episode_from_events(
            args.target_ip, run_id, events, match_all=args.match_all
        )
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    ep_dir = REPO_ROOT / "outputs" / "episodes" / "wazuh_mvp"
    ep_dir.mkdir(parents=True, exist_ok=True)
    ep_path = ep_dir / f"{episode_id}.json"
    ep_path.write_text(episode.model_dump_json(indent=2), encoding="utf-8")
    print(f"Episode written: {ep_path.relative_to(REPO_ROOT)} (alerts={len(events)})")

    if args.dry_run:
        print("dry-run: 略過 retrieve / analyze / writeback / DB")
        return 0

    status, writeback_payload, evidence_payload, layerc_summary, agent_payload, issues_payload = (
        run_layer_c_pipeline(
            episode,
            ep_path,
            episode_id,
            repo_root=REPO_ROOT,
            do_writeback=not args.no_writeback,
            writeback_mode=args.writeback_mode,
            triage_rules=triage_rules,
        )
    )

    print(
        "Layer C: "
        f"C1={layerc_summary.get('c1_retrieve')} "
        f"C2={layerc_summary.get('c2_analyze')} "
        f"C3={layerc_summary.get('c3_writeback')}"
    )

    row = {
        "source": "wazuh",
        "dataset_label": None,
        "episode_id": episode_id,
        "run_id": run_id,
        "target_ip": args.target_ip,
        "window_start_ms": episode.t0_ms,
        "window_end_ms": episode.t1_ms,
        "alert_count": len(events),
        "status": status,
        "evidence_json": evidence_payload,
        "agent_outputs_json": agent_payload,
        "issues_json": issues_payload,
        "error_message": None if status == "ok" else "validation_failed_or_partial",
        "writeback_json": writeback_payload,
        "layerc_summary": layerc_summary,
    }

    if args.skip_db:
        print(json.dumps({k: v for k, v in row.items() if k != "evidence_json"}, indent=2, default=str))
        print("skip-db: 未寫入 PostgreSQL")
        return 0 if status == "ok" else 1

    if not database_url:
        print("未設定 DATABASE_URL，無法寫入 DB（可用 --skip-db）", file=sys.stderr)
        return 1

    try:
        new_id = insert_analysis_run(database_url, row)
    except Exception as e:
        print(f"PostgreSQL insert failed: {e}", file=sys.stderr)
        return 1

    print(f"PostgreSQL analysis_runs.id = {new_id} status={status}")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
