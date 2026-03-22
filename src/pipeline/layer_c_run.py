"""
Shared Layer C 管線：C1 retrieve → C2 analyze →（可選）C3 writeback。
供 mvp_wazuh_episode_pg、mvp_cert_layer_c、HTTP API 共用。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from src.contracts.episode import Episode
from src.pipeline.triage_rules import resolve_triage_rules


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_layer_c_pipeline(
    episode: Any,
    ep_path: Path,
    episode_id: str,
    *,
    repo_root: Optional[Path] = None,
    do_writeback: bool = True,
    writeback_mode: str = "dry_run",
    triage_rules: Optional[list[str]] = None,
) -> tuple[
    str,
    Optional[dict[str, Any]],
    Optional[dict[str, Any]],
    dict[str, Any],
    Optional[dict[str, Any]],
    Optional[dict[str, Any]],
]:
    """
    C1 retrieve → C2 analyze →（可選）C3 writeback。

    回傳 (analyze_status, writeback_payload, evidence_payload, layerc_summary, agent_payload, issues_payload)
    """
    root = repo_root or default_repo_root()
    from src.pipeline.retrieve_evidence import build_evidence_set
    from src.pipeline.validate_and_repair import run_analyze_by_rules

    summary: dict[str, Any] = {
        "c1_retrieve": "pending",
        "c2_analyze": "pending",
        "c3_writeback": "skipped",
        "paths": {},
    }

    evidence_set_obj = build_evidence_set(ep_path)
    evidence_path = root / "outputs" / "evidence" / f"{episode_id}.json"
    summary["c1_retrieve"] = "ok" if evidence_path.exists() else "failed"
    summary["paths"]["evidence"] = str(evidence_path.relative_to(root))

    evidence_payload: Optional[dict[str, Any]] = None
    if evidence_path.exists():
        evidence_payload = json.loads(evidence_path.read_text(encoding="utf-8"))

    ep_model = episode if isinstance(episode, Episode) else Episode.model_validate(episode)
    rules = resolve_triage_rules(ep_model, explicit=triage_rules)
    summary["triage_rules"] = list(rules)
    summary["rules_count"] = len(rules)

    agent_payload, status = run_analyze_by_rules(ep_model, evidence_set_obj, rules, trust_signals=None)
    summary["c2_analyze"] = status
    summary["paths"]["agents_dir"] = str((root / "outputs" / "agents").relative_to(root))
    if status == "ok":
        summary["paths"]["agents_by_rule"] = f"outputs/agents/{episode_id}_by_rule.json"

    issues_payload: Optional[dict[str, Any]] = None
    issues_path = root / "outputs" / "issues" / f"{episode_id}.json"
    if issues_path.exists():
        try:
            issues_payload = json.loads(issues_path.read_text(encoding="utf-8"))
        except Exception:
            issues_payload = {"path": str(issues_path)}

    writeback_payload: Optional[dict[str, Any]] = None
    if do_writeback and status == "ok":
        from src.pipeline.writeback_pipeline import run_writeback

        try:
            _patch, wb_path = run_writeback(str(ep_path), mode=writeback_mode)
            summary["c3_writeback"] = "ok"
            summary["paths"]["writeback"] = str(wb_path.relative_to(root))
            summary["writeback_mode"] = writeback_mode
            if wb_path.exists():
                writeback_payload = json.loads(wb_path.read_text(encoding="utf-8"))
        except Exception as e:
            summary["c3_writeback"] = "failed"
            summary["c3_writeback_error"] = str(e)[:800]
    elif do_writeback and status != "ok":
        summary["c3_writeback"] = "skipped_analyze_failed"

    agent_out = agent_payload if status == "ok" else None
    return status, writeback_payload, evidence_payload, summary, agent_out, issues_payload
