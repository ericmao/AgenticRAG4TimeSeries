"""
Validate agent outputs (citations + policy guardrails). Repair loop: rerun failing agents up to 2 times with repair_hint.
Write outputs/agents/ only when all valid; on final failure write outputs/issues/<episode_id>.json. Audit validation results.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from src.contracts.agent_output import AgentOutput
from src.contracts.episode import Episode
from src.contracts.evidence import EvidenceSet
from src.pipeline.run_agents import run_all_agents
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
        # Rerun only failing agents with repair_hint
        for agent_id in list(errors_by_agent.keys()):
            runners = {
                "triage": run_all_agents,
                "hunt_planner": run_all_agents,
                "response_advisor": run_all_agents,
            }
            if agent_id not in runners:
                continue
            new_outputs = run_all_agents(
                episode,
                evidence_set,
                trust_signals=trust_signals,
                repair_hint=REPAIR_HINT,
                write_outputs=False,
            )
            outputs[agent_id] = new_outputs[agent_id]
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


def run_analyze_with_validation(
    episode: Episode,
    evidence_set: EvidenceSet,
    trust_signals: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, AgentOutput], str]:
    """
    Run agents (no write), then validate_and_repair. Write outputs/agents/ only when status=="ok";
    on "failed" issues already written by validate_and_repair. Returns (outputs, status).
    """
    root = _repo_root()
    out_dir = root / "outputs" / "agents"
    episode_id = episode.episode_id
    initial = run_all_agents(episode, evidence_set, trust_signals, repair_hint=None, write_outputs=False)
    outputs, status = validate_and_repair(episode, evidence_set, initial, trust_signals=trust_signals)
    if status == "ok":
        out_dir.mkdir(parents=True, exist_ok=True)
        for agent_id, out in outputs.items():
            path = out_dir / f"{episode_id}_{agent_id}.json"
            path.write_text(out.model_dump_json(indent=2), encoding="utf-8")
    return outputs, status
