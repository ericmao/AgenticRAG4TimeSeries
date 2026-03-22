#!/usr/bin/env python3
"""
CERT / demo Episode → Layer C 完整管線 → PostgreSQL analysis_runs

用法:
  PYTHONPATH=. python scripts/mvp_cert_layer_c.py --episode tests/demo/episode_insider_highrisk.json
  PYTHONPATH=. python scripts/mvp_cert_layer_c.py --episode outputs/episodes/cert/foo.json --skip-db
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from src.contracts.episode import Episode
from src.pipeline.layer_c_run import run_layer_c_pipeline
from src.storage.analysis_runs_store import insert_analysis_run


def main() -> int:
    parser = argparse.ArgumentParser(description="CERT Episode → Layer C → analysis_runs")
    parser.add_argument("--episode", required=True, help="Episode JSON 路徑（相對或絕對）")
    parser.add_argument("--dataset-label", default="cert", help="寫入 dataset_label")
    parser.add_argument("--dry-run", action="store_true", help="只印 episode 資訊，不跑 Layer C、不寫 DB")
    parser.add_argument("--skip-db", action="store_true", help="跑分析但不寫 PostgreSQL")
    parser.add_argument("--no-writeback", action="store_true", help="略過 C3 writeback")
    parser.add_argument("--writeback-mode", default="dry_run", help="dry_run | review | auto")
    parser.add_argument(
        "--triage-rules",
        default=None,
        help="逗號分隔 rule_id（如 lateral,burst）；未指定則用 TRIAGE_RULES 或 episode.sequence_tags",
    )
    args = parser.parse_args()

    triage_rules = None
    if args.triage_rules and str(args.triage_rules).strip():
        triage_rules = [x.strip() for x in str(args.triage_rules).split(",") if x.strip()]

    ep_path = Path(args.episode)
    if not ep_path.is_absolute():
        ep_path = REPO_ROOT / ep_path
    if not ep_path.exists():
        print(f"Episode not found: {ep_path}", file=sys.stderr)
        return 1

    raw = json.loads(ep_path.read_text(encoding="utf-8"))
    episode = Episode.model_validate(raw)
    episode_id = episode.episode_id
    run_id = f"cert-{uuid.uuid4().hex[:16]}"

    if args.dry_run:
        print(f"episode_id={episode_id} run_id={run_id} path={ep_path.relative_to(REPO_ROOT)}")
        print("dry-run: 略過 Layer C / DB")
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

    events = episode.events or []
    row = {
        "source": "cert",
        "dataset_label": args.dataset_label,
        "episode_id": episode_id,
        "run_id": run_id,
        "target_ip": None,
        "window_start_ms": int(episode.t0_ms),
        "window_end_ms": int(episode.t1_ms),
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

    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        print("未設定 DATABASE_URL（可用 --skip-db）", file=sys.stderr)
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
