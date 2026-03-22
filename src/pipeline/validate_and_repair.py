"""
Validate agent outputs (citations + policy guardrails). Repair loop: rerun failing agents up to 2 times with repair_hint.
Write outputs/agents/ only when all valid; on final failure write outputs/issues/<episode_id>.json. Audit validation results.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

from src.contracts.agent_output import AgentOutput
from src.contracts.episode import Episode
from src.contracts.evidence import EvidenceSet
from src.pipeline.run_agents import run_agents_for_rule, run_single_agent
from src.validators.citations import validate_citations
from src.validators.policy_guardrails import validate_response_advisor_policy
from src.utils.audit_logger import audit_log

REPAIR_HINT = (
    "Fix the issues: add valid evidence_id citations; make targets specific using episode entities/artifacts; keep actions conservative."
)
MAX_REPAIR_ATTEMPTS = 2


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _validate_all(
    outputs: dict[str, AgentOutput],
    episode: Episode,
    evidence_set: EvidenceSet,
    min_citations: int = 3,
) -> tuple[dict[str, AgentOutput], dict[str, list[str]]]:
    """Run citation + policy validators. Return (outputs, agent_id -> list of error strings)."""
    errors_by_agent: dict[str, list[str]] = {}
    for agent_id, out in outputs.items():
        errs: list[str] = []
        cr = validate_citations(out, evidence_set, min_citations=min_citations)
        if not cr["valid"]:
            errs.extend(cr.get("errors") or [])
        if agent_id == "response_advisor":
            pr = validate_response_advisor_policy(out, episode)
            if not pr["valid"]:
                errs.extend(pr.get("errors") or [])
        if errs:
            errors_by_agent[agent_id] = errs
    return outputs, errors_by_agent


def validate_and_repair(
    episode: Episode,
    evidence_set: EvidenceSet,
    initial_outputs: dict[str, AgentOutput],
    trust_signals: Optional[dict[str, Any]] = None,
    min_citations: int = 3,
) -> tuple[dict[str, AgentOutput], str]:
    """
    Validate initial_outputs; if any fail, rerun failing agents up to MAX_REPAIR_ATTEMPTS with REPAIR_HINT.
    Returns (final_outputs, status) where status is "ok" or "failed".
    Does not write to disk; caller writes outputs/agents/ when status=="ok" or outputs/issues/ when "failed".
    """
    root = _repo_root()
    run_id = episode.run_id
    episode_id = episode.episode_id
    from src.config import get_config
    cfg = get_config()
    prompt_version = getattr(cfg, "PROMPT_VERSION", "v0.1")

    outputs = dict(initial_outputs)
    attempts = 0
    while attempts <= MAX_REPAIR_ATTEMPTS:
        _, errors_by_agent = _validate_all(outputs, episode, evidence_set, min_citations)
        if not errors_by_agent:
            audit_log(
                event={"action": "validation_passed", "attempt": attempts},
                run_id=run_id,
                episode_id=episode_id,
                component="pipeline.validate_and_repair",
                prompt_version=prompt_version,
            )
            return outputs, "ok"
        audit_log(
            event={"action": "validation_failed", "errors_by_agent": errors_by_agent, "attempt": attempts},
            run_id=run_id,
            episode_id=episode_id,
            component="pipeline.validate_and_repair",
            prompt_version=prompt_version,
        )
        if attempts == MAX_REPAIR_ATTEMPTS:
            break
        # Rerun only failing agents with repair_hint（單一 default 規則）
        for agent_id in list(errors_by_agent.keys()):
            if agent_id not in ("triage", "hunt_planner", "response_advisor"):
                continue
            outputs[agent_id] = run_single_agent(
                agent_id,
                episode,
                evidence_set,
                trust_signals=trust_signals,
                repair_hint=REPAIR_HINT,
                write_outputs=False,
                rule_id="default",
            )
        attempts += 1

    # Final failure: write issues
    _, errors_by_agent = _validate_all(outputs, episode, evidence_set, min_citations)
    issues_dir = root / "outputs" / "issues"
    issues_dir.mkdir(parents=True, exist_ok=True)
    issues_path = issues_dir / f"{episode_id}.json"
    payload = {
        "status": "failed",
        "episode_id": episode_id,
        "run_id": run_id,
        "errors_by_agent": errors_by_agent,
        "last_outputs": {aid: out.model_dump() for aid, out in outputs.items()},
    }
    issues_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    audit_log(
        event={"action": "validation_failed_final", "issues_path": str(issues_path.relative_to(root))},
        run_id=run_id,
        episode_id=episode_id,
        component="pipeline.validate_and_repair",
        prompt_version=prompt_version,
    )
    return outputs, "failed"


def validate_and_repair_rule_bundle(
    episode: Episode,
    evidence_set: EvidenceSet,
    initial_outputs: dict[str, AgentOutput],
    rule_id: str,
    trust_signals: Optional[dict[str, Any]] = None,
    min_citations: int = 3,
) -> tuple[dict[str, AgentOutput], str]:
    """
    驗證單一規則的三個代理輸出；失敗時僅重跑該規則內失敗的代理。不寫 issues（由 run_analyze_by_rules 統一寫）。
    """
    run_id = episode.run_id
    episode_id = episode.episode_id
    from src.config import get_config

    cfg = get_config()
    prompt_version = getattr(cfg, "PROMPT_VERSION", "v0.1")

    outputs = dict(initial_outputs)
    attempts = 0
    while attempts <= MAX_REPAIR_ATTEMPTS:
        _, errors_by_agent = _validate_all(outputs, episode, evidence_set, min_citations)
        if not errors_by_agent:
            audit_log(
                event={"action": "validation_passed_rule", "attempt": attempts, "rule_id": rule_id},
                run_id=run_id,
                episode_id=episode_id,
                component="pipeline.validate_and_repair",
                prompt_version=prompt_version,
            )
            return outputs, "ok"
        audit_log(
            event={
                "action": "validation_failed_rule",
                "errors_by_agent": errors_by_agent,
                "attempt": attempts,
                "rule_id": rule_id,
            },
            run_id=run_id,
            episode_id=episode_id,
            component="pipeline.validate_and_repair",
            prompt_version=prompt_version,
        )
        if attempts == MAX_REPAIR_ATTEMPTS:
            break
        for agent_id in list(errors_by_agent.keys()):
            if agent_id not in ("triage", "hunt_planner", "response_advisor"):
                continue
            outputs[agent_id] = run_single_agent(
                agent_id,
                episode,
                evidence_set,
                trust_signals=trust_signals,
                repair_hint=REPAIR_HINT,
                write_outputs=False,
                rule_id=rule_id,
            )
        attempts += 1

    return outputs, "failed"


def _write_agent_disk_by_rule(
    episode_id: str,
    by_rule: dict[str, dict[str, AgentOutput]],
    root: Path,
) -> None:
    """寫入 _by_rule.json 與第一規則之 triage/hunt_planner/response_advisor 三檔（相容舊工具）。"""
    out_dir = root / "outputs" / "agents"
    out_dir.mkdir(parents=True, exist_ok=True)
    serial: dict[str, Any] = {}
    for rid, bundle in by_rule.items():
        serial[rid] = {k: v.model_dump() for k, v in bundle.items()}
    by_path = out_dir / f"{episode_id}_by_rule.json"
    by_path.write_text(json.dumps(serial, indent=2, ensure_ascii=False), encoding="utf-8")

    first_rid = next(iter(by_rule))
    first = by_rule[first_rid]
    for agent_id in ("triage", "hunt_planner", "response_advisor"):
        p = out_dir / f"{episode_id}_{agent_id}.json"
        p.write_text(first[agent_id].model_dump_json(indent=2), encoding="utf-8")


def _write_multi_rule_issues(
    episode: Episode,
    rule_id: str,
    errors_by_agent: dict[str, list[str]],
    last_outputs: dict[str, AgentOutput],
    root: Path,
) -> None:
    issues_dir = root / "outputs" / "issues"
    issues_dir.mkdir(parents=True, exist_ok=True)
    episode_id = episode.episode_id
    issues_path = issues_dir / f"{episode_id}.json"
    payload = {
        "status": "failed",
        "multi_rule": True,
        "failed_rule": rule_id,
        "episode_id": episode_id,
        "run_id": episode.run_id,
        "errors_by_agent": errors_by_agent,
        "last_outputs": {aid: out.model_dump() for aid, out in last_outputs.items()},
    }
    issues_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    from src.config import get_config as _gc

    _cfg = _gc()
    audit_log(
        event={"action": "validation_failed_final_rule", "issues_path": str(issues_path.relative_to(root)), "rule_id": rule_id},
        run_id=episode.run_id,
        episode_id=episode_id,
        component="pipeline.validate_and_repair",
        prompt_version=getattr(_cfg, "PROMPT_VERSION", "v0.1"),
    )


def run_analyze_by_rules(
    episode: Episode,
    evidence_set: EvidenceSet,
    rules: Sequence[str],
    trust_signals: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], str]:
    """
    對每個 rule_id 跑 triage→hunt→response，各別驗證修復。
    回傳 (agent_payload_for_db, status)；status 為 ok 時已寫入 outputs/agents/*。
    agent_payload 含 by_rule、triage/hunt_planner/response_advisor（第一規則複本）。
    """
    root = _repo_root()
    episode_id = episode.episode_id
    by_rule: dict[str, dict[str, AgentOutput]] = {}

    for rule_id in rules:
        initial = run_agents_for_rule(
            episode,
            evidence_set,
            rule_id,
            trust_signals=trust_signals,
            repair_hint=None,
            write_outputs=False,
        )
        outputs, status = validate_and_repair_rule_bundle(
            episode,
            evidence_set,
            initial,
            rule_id,
            trust_signals=trust_signals,
        )
        if status != "ok":
            _, errors_by_agent = _validate_all(outputs, episode, evidence_set, min_citations=3)
            _write_multi_rule_issues(episode, rule_id, errors_by_agent, outputs, root)
            return {}, "failed"
        by_rule[rule_id] = outputs

    _write_agent_disk_by_rule(episode_id, by_rule, root)

    first_rid = next(iter(by_rule))
    first = by_rule[first_rid]
    agent_payload: dict[str, Any] = {
        "by_rule": {rid: {k: v.model_dump() for k, v in bundle.items()} for rid, bundle in by_rule.items()},
        "triage": first["triage"].model_dump(),
        "hunt_planner": first["hunt_planner"].model_dump(),
        "response_advisor": first["response_advisor"].model_dump(),
    }
    return agent_payload, "ok"


def run_analyze_with_validation(
    episode: Episode,
    evidence_set: EvidenceSet,
    trust_signals: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, AgentOutput], str]:
    """
    單一 default 規則（舊 API）：回傳 (扁平 triage/hunt_planner/response_advisor, status)。
    內部改為 run_analyze_by_rules(["default"])，再還原扁平 dict。
    """
    payload, status = run_analyze_by_rules(episode, evidence_set, ["default"], trust_signals=trust_signals)
    if status != "ok":
        return {}, "failed"
    # 由 by_rule 還原 AgentOutput
    br = payload["by_rule"]["default"]
    flat = {
        "triage": AgentOutput.model_validate(br["triage"]),
        "hunt_planner": AgentOutput.model_validate(br["hunt_planner"]),
        "response_advisor": AgentOutput.model_validate(br["response_advisor"]),
    }
    return flat, "ok"
