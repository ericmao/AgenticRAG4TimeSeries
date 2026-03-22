#!/usr/bin/env python3
"""
將 **analysis_runs** 從來源 PostgreSQL 合併至目標 DB（保留 id，供 /runs、/investigations/graph?run_id= 一致）。

典型情境：本機 Docker（localhost:5435）已有 r3.2／CERT／Wazuh 跑出的紀錄，要同步到 192.168.1.203 等同架構。

用法:
  export SOURCE_DATABASE_URL='postgresql://agentic:agentic@127.0.0.1:5435/agentic'
  export TARGET_DATABASE_URL='postgresql://agentic:agentic@192.168.1.203:5435/agentic'
  PYTHONPATH=. python scripts/migrate_analysis_runs_pg.py --dry-run
  PYTHONPATH=. python scripts/migrate_analysis_runs_pg.py

選項:
  --source-only cert|wazuh   只遷移該 source（預設：全部）
  --ids 3,4,14               只遷移列出的 id（逗號分隔）
  --dry-run                  只印筆數與 id，不寫入

注意:
  - 目標須已存在 analysis_runs 表（docker compose 啟動 MVP 時會自動建）。
  - 若 id 已存在，預設 **覆寫** 該列（ON CONFLICT DO UPDATE）；可用 --skip-existing 改為跳過。
  - 遷移後會 `setval` 序號，避免之後 INSERT 撞 id。

「三圖」空白：多為該筆 **agent_outputs_json** 為空或未完成 C2；若本機有圖、遷移後仍應有圖。
若 iframe 整片白，請在遠端確認已 rsync **services/mvp_ui_api/static/investigation/** 並重啟 mvp_ui。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass


def _connect(url: str):
    import psycopg2

    return psycopg2.connect(url)


def _fetch_rows(
    cur,
    *,
    source_only: Optional[str],
    ids: Optional[list[int]],
) -> list[dict[str, Any]]:
    if ids:
        cur.execute(
            """
            SELECT id, created_at, source, dataset_label, episode_id, run_id, target_ip,
                   window_start_ms, window_end_ms, alert_count, status,
                   evidence_json, agent_outputs_json, issues_json, error_message,
                   writeback_json, layerc_summary
            FROM analysis_runs
            WHERE id = ANY(%s)
            ORDER BY id
            """,
            (ids,),
        )
    else:
        q = """
            SELECT id, created_at, source, dataset_label, episode_id, run_id, target_ip,
                   window_start_ms, window_end_ms, alert_count, status,
                   evidence_json, agent_outputs_json, issues_json, error_message,
                   writeback_json, layerc_summary
            FROM analysis_runs
        """
        params: tuple[Any, ...] = ()
        if source_only:
            q += " WHERE source = %s"
            params = (source_only,)
        q += " ORDER BY id"
        cur.execute(q, params)
    cols = [d[0] for d in cur.description]
    out = []
    for row in cur.fetchall():
        out.append(dict(zip(cols, row)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Migrate analysis_runs between Postgres instances.")
    ap.add_argument("--source-url", default=os.environ.get("SOURCE_DATABASE_URL", ""))
    ap.add_argument("--target-url", default=os.environ.get("TARGET_DATABASE_URL", ""))
    ap.add_argument("--source-only", choices=("cert", "wazuh"), default=None)
    ap.add_argument("--ids", default="", help="Comma-separated run ids, e.g. 3,4,14")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="若目標已有同 id，不覆寫",
    )
    args = ap.parse_args()
    if not args.source_url or not args.target_url:
        print("需要 --source-url / --target-url 或環境變數 SOURCE_DATABASE_URL / TARGET_DATABASE_URL", file=sys.stderr)
        return 1

    id_list: Optional[list[int]] = None
    if args.ids.strip():
        id_list = [int(x.strip()) for x in args.ids.split(",") if x.strip()]

    src = _connect(args.source_url)
    tgt = _connect(args.target_url)
    try:
        from psycopg2.extras import Json

        sc = src.cursor()
        rows = _fetch_rows(sc, source_only=args.source_only, ids=id_list)
        print(f"source rows: {len(rows)}")
        if args.dry_run:
            for r in rows:
                print(f"  id={r['id']} source={r['source']} episode_id={r['episode_id']!r}")
            return 0

        tc = tgt.cursor()
        sql_upsert = """
            INSERT INTO analysis_runs (
                id, created_at, source, dataset_label, episode_id, run_id, target_ip,
                window_start_ms, window_end_ms, alert_count, status,
                evidence_json, agent_outputs_json, issues_json, error_message,
                writeback_json, layerc_summary
            ) VALUES (
                %(id)s, %(created_at)s, %(source)s, %(dataset_label)s, %(episode_id)s, %(run_id)s, %(target_ip)s,
                %(window_start_ms)s, %(window_end_ms)s, %(alert_count)s, %(status)s,
                %(evidence_json)s, %(agent_outputs_json)s, %(issues_json)s, %(error_message)s,
                %(writeback_json)s, %(layerc_summary)s
            )
            ON CONFLICT (id) DO UPDATE SET
                created_at = EXCLUDED.created_at,
                source = EXCLUDED.source,
                dataset_label = EXCLUDED.dataset_label,
                episode_id = EXCLUDED.episode_id,
                run_id = EXCLUDED.run_id,
                target_ip = EXCLUDED.target_ip,
                window_start_ms = EXCLUDED.window_start_ms,
                window_end_ms = EXCLUDED.window_end_ms,
                alert_count = EXCLUDED.alert_count,
                status = EXCLUDED.status,
                evidence_json = EXCLUDED.evidence_json,
                agent_outputs_json = EXCLUDED.agent_outputs_json,
                issues_json = EXCLUDED.issues_json,
                error_message = EXCLUDED.error_message,
                writeback_json = EXCLUDED.writeback_json,
                layerc_summary = EXCLUDED.layerc_summary
        """
        sql_insert_only = """
            INSERT INTO analysis_runs (
                id, created_at, source, dataset_label, episode_id, run_id, target_ip,
                window_start_ms, window_end_ms, alert_count, status,
                evidence_json, agent_outputs_json, issues_json, error_message,
                writeback_json, layerc_summary
            ) VALUES (
                %(id)s, %(created_at)s, %(source)s, %(dataset_label)s, %(episode_id)s, %(run_id)s, %(target_ip)s,
                %(window_start_ms)s, %(window_end_ms)s, %(alert_count)s, %(status)s,
                %(evidence_json)s, %(agent_outputs_json)s, %(issues_json)s, %(error_message)s,
                %(writeback_json)s, %(layerc_summary)s
            )
            ON CONFLICT (id) DO NOTHING
        """

        n_ok = 0
        n_skip = 0
        for r in rows:
            payload = {
                "id": r["id"],
                "created_at": r["created_at"],
                "source": r["source"],
                "dataset_label": r["dataset_label"],
                "episode_id": r["episode_id"],
                "run_id": r["run_id"],
                "target_ip": r["target_ip"],
                "window_start_ms": r["window_start_ms"],
                "window_end_ms": r["window_end_ms"],
                "alert_count": r["alert_count"],
                "status": r["status"],
                "evidence_json": Json(r["evidence_json"]) if r["evidence_json"] is not None else None,
                "agent_outputs_json": Json(r["agent_outputs_json"]) if r["agent_outputs_json"] is not None else None,
                "issues_json": Json(r["issues_json"]) if r["issues_json"] is not None else None,
                "error_message": r["error_message"],
                "writeback_json": Json(r["writeback_json"]) if r["writeback_json"] is not None else None,
                "layerc_summary": Json(r["layerc_summary"]) if r["layerc_summary"] is not None else None,
            }
            if args.skip_existing:
                tc.execute(sql_insert_only, payload)
                if tc.rowcount == 0:
                    n_skip += 1
                else:
                    n_ok += 1
            else:
                tc.execute(sql_upsert, payload)
                n_ok += 1

        tgt.commit()
        tc.execute(
            "SELECT setval(pg_get_serial_sequence('analysis_runs', 'id'), "
            "COALESCE((SELECT MAX(id) FROM analysis_runs), 1))"
        )
        tgt.commit()
        print(f"done: applied={n_ok} skipped={n_skip} (skip-existing={args.skip_existing})")
    finally:
        src.close()
        tgt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
