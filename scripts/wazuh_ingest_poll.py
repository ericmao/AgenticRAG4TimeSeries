#!/usr/bin/env python3
"""
Wazuh Indexer 輪詢：依 wazuh_ingest_state 游標拉取新 alerts → Episode → Layer C → analysis_runs。

環境變數：WAZUH_INDEXER_URL、WAZUH_INDEXER_USERNAME、WAZUH_INDEXER_PASSWORD、WAZUH_VERIFY_SSL、DATABASE_URL

自動 episode 準則（可選，macOS／Windows 皆透過環境變數或 --auto-criteria）：
  WAZUH_AUTO_EPISODE_CRITERIA=1  啟用 OR 條件（亦可用 --auto-criteria）
  WAZUH_AUTO_MIN_RULE_LEVEL=12   單筆等級 ≥ 此值則通過（0=關閉此條）
  WAZUH_AUTO_MIN_ALERT_COUNT=5   本批筆數 ≥ 此值（0=關閉）
  WAZUH_AUTO_MIN_DISTINCT_RULES=3  不同 rule.id 數（0=關閉）
  WAZUH_AUTO_MITRE_TRIGGERS=1    任一有 MITRE 則通過
  WAZUH_AUTO_DEDUP_MINUTES=60    同一 target_ip 最短間隔分鐘（0=關閉；仍會前進游標）

用法:
  PYTHONPATH=. python scripts/wazuh_ingest_poll.py --target-ip 192.168.1.203 --match-all --once
  PYTHONPATH=. python scripts/wazuh_ingest_poll.py --target-ip 10.0.0.5 --interval-sec 300
  PYTHONPATH=. python scripts/wazuh_ingest_poll.py --target-ip 10.0.0.5 --auto-criteria --once
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from src.integrations.wazuh_env import apply_wazuh_env_from_dotenv

apply_wazuh_env_from_dotenv(REPO_ROOT)

from src.integrations.wazuh_auto_criteria import (
    WazuhAutoCriteria,
    dedup_minutes_from_env,
    dedup_should_skip_last_run,
    should_create_episode,
)
from src.integrations.wazuh_indexer_client import (
    build_episode_from_events,
    build_ip_query,
    build_ip_query_after_ms,
    hit_to_event,
    indexer_post,
    max_event_ts_ms,
)
from src.pipeline.layer_c_run import run_evidenceops_layer_c_pipeline, run_layer_c_pipeline
from src.storage.analysis_runs_store import insert_analysis_run
from src.storage.ingest_state_store import get_state, upsert_state


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def run_poll_cycle(
    *,
    target_ip: str,
    match_all: bool,
    size: int,
    hours_fallback: int,
    writeback_mode: str,
    no_writeback: bool,
    skip_db: bool,
    key_name: str,
    triage_rules: list[str] | None = None,
    evidenceops: bool = False,
    auto_criteria: bool = False,
) -> int:
    base = os.environ.get("WAZUH_INDEXER_URL", "").strip()
    user = os.environ.get("WAZUH_INDEXER_USERNAME", "").strip()
    password = os.environ.get("WAZUH_INDEXER_PASSWORD", "")
    verify_ssl = _bool_env("WAZUH_VERIFY_SSL", default=False)
    database_url = os.environ.get("DATABASE_URL", "").strip()

    if not base or not user:
        print("請設定 WAZUH_INDEXER_URL、WAZUH_INDEXER_USERNAME、WAZUH_INDEXER_PASSWORD", file=sys.stderr)
        return 1

    last_ms = 0
    if database_url:
        st = get_state(database_url, key_name)
        last_ms = int(st.get("last_timestamp_ms") or 0)

    if last_ms > 0:
        query_body = build_ip_query_after_ms(target_ip, last_ms, size, match_all=match_all)
    else:
        query_body = build_ip_query(target_ip, hours_fallback, size, match_all=match_all)

    try:
        raw = indexer_post(
            base,
            "/wazuh-alerts-*/_search",
            query_body,
            user,
            password,
            verify_ssl,
        )
    except Exception as e:
        print(f"Indexer error: {e}", file=sys.stderr)
        return 1

    hits = (raw.get("hits") or {}).get("hits") or []
    events = [hit_to_event(h) for h in hits]
    if not events:
        print("poll: no new alerts")
        return 0

    new_max = max_event_ts_ms(events)

    dedup_min = dedup_minutes_from_env()
    if database_url and dedup_min > 0:
        from src.storage.analysis_runs_store import get_latest_wazuh_created_at_for_target

        last_at = get_latest_wazuh_created_at_for_target(database_url, target_ip)
        if dedup_should_skip_last_run(last_at, dedup_min):
            if new_max > last_ms:
                upsert_state(database_url, key_name, last_timestamp_ms=new_max)
            print(f"poll: dedup skip target_ip={target_ip} within {dedup_min}m (cursor advanced)")
            return 0

    crit = WazuhAutoCriteria.from_env()
    if auto_criteria:
        crit = replace(crit, enabled=True)
    if crit.enabled:
        ok, crit_detail = should_create_episode(events, crit)
        if not ok:
            if database_url and new_max > last_ms:
                upsert_state(database_url, key_name, last_timestamp_ms=new_max)
            print(f"poll: auto criteria skip {crit_detail}")
            return 0

    run_id = f"wazuh-poll-{uuid.uuid4().hex[:16]}"
    episode, episode_id = build_episode_from_events(target_ip, run_id, events, match_all=match_all)

    ep_dir = REPO_ROOT / "outputs" / "episodes" / "wazuh_mvp"
    ep_dir.mkdir(parents=True, exist_ok=True)
    ep_path = ep_dir / f"{episode_id}.json"
    ep_path.write_text(episode.model_dump_json(indent=2), encoding="utf-8")

    if evidenceops:
        status, writeback_payload, evidence_payload, layerc_summary, agent_payload, issues_payload = (
            run_evidenceops_layer_c_pipeline(
                episode,
                ep_path,
                episode_id,
                repo_root=REPO_ROOT,
                do_writeback=not no_writeback,
                writeback_mode=writeback_mode,
                triage_rules=triage_rules,
            )
        )
    else:
        status, writeback_payload, evidence_payload, layerc_summary, agent_payload, issues_payload = (
            run_layer_c_pipeline(
                episode,
                ep_path,
                episode_id,
                repo_root=REPO_ROOT,
                do_writeback=not no_writeback,
                writeback_mode=writeback_mode,
                triage_rules=triage_rules,
            )
        )

    if database_url and new_max > last_ms:
        upsert_state(database_url, key_name, last_timestamp_ms=new_max)

    row = {
        "source": "wazuh",
        "dataset_label": "ingest_poll",
        "episode_id": episode_id,
        "run_id": run_id,
        "target_ip": target_ip,
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

    if skip_db or not database_url:
        print(f"poll ok episode_id={episode_id} status={status} (skip-db)")
        return 0 if status == "ok" else 1

    try:
        rid = insert_analysis_run(database_url, row)
        print(f"poll ok analysis_runs.id={rid} episode_id={episode_id} status={status}")
    except Exception as e:
        print(f"DB insert failed: {e}", file=sys.stderr)
        return 1

    return 0 if status == "ok" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Wazuh ingest poller with cursor")
    parser.add_argument("--target-ip", required=True)
    parser.add_argument("--match-all", action="store_true")
    parser.add_argument("--size", type=int, default=500)
    parser.add_argument("--hours-fallback", type=int, default=24, help="首輪無游標時使用 now-Nh")
    parser.add_argument("--interval-sec", type=int, default=300)
    parser.add_argument("--once", action="store_true", help="只跑一輪")
    parser.add_argument("--no-writeback", action="store_true")
    parser.add_argument("--writeback-mode", default="dry_run")
    parser.add_argument("--skip-db", action="store_true")
    parser.add_argument(
        "--state-key",
        default=None,
        help="wazuh_ingest_state.key_name，預設 wazuh_poll:<ip>:<match_all>",
    )
    parser.add_argument(
        "--triage-rules",
        default=None,
        help="逗號分隔 rule_id；未指定則用 TRIAGE_RULES 或 episode.sequence_tags",
    )
    parser.add_argument(
        "--evidenceops",
        action="store_true",
        help="走 EvidenceOps 多代理（CaseOrchestrator）",
    )
    parser.add_argument(
        "--auto-criteria",
        action="store_true",
        help="啟用自動 episode 行為準則（等同 WAZUH_AUTO_EPISODE_CRITERIA=1，門檻見環境變數）",
    )
    args = parser.parse_args()

    triage_rules = None
    if args.triage_rules and str(args.triage_rules).strip():
        triage_rules = [x.strip() for x in str(args.triage_rules).split(",") if x.strip()]

    key = args.state_key or f"wazuh_poll:{args.target_ip}:{'all' if args.match_all else 'ip'}"

    if args.once:
        return run_poll_cycle(
            target_ip=args.target_ip,
            match_all=args.match_all,
            size=args.size,
            hours_fallback=args.hours_fallback,
            writeback_mode=args.writeback_mode,
            no_writeback=args.no_writeback,
            skip_db=args.skip_db,
            key_name=key,
            triage_rules=triage_rules,
            evidenceops=args.evidenceops,
            auto_criteria=args.auto_criteria,
        )

    while True:
        rc = run_poll_cycle(
            target_ip=args.target_ip,
            match_all=args.match_all,
            size=args.size,
            hours_fallback=args.hours_fallback,
            writeback_mode=args.writeback_mode,
            no_writeback=args.no_writeback,
            skip_db=args.skip_db,
            key_name=key,
            triage_rules=triage_rules,
            evidenceops=args.evidenceops,
            auto_criteria=args.auto_criteria,
        )
        if rc != 0:
            print(f"poll cycle exit {rc}", file=sys.stderr)
        time.sleep(max(1, args.interval_sec))


if __name__ == "__main__":
    raise SystemExit(main())
