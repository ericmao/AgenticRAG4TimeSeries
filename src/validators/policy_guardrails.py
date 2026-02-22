"""
Policy guardrails for ResponseAdvisor: action allowlist, specific targets, duration/guardrails/rollback.
"""
from __future__ import annotations

import re
from typing import Any

from src.contracts.agent_output import AgentOutput
from src.contracts.episode import Episode

ACTION_ALLOWLIST = {"block", "isolate", "watchlist", "collect_more_data"}
VAGUE_TARGET_PHRASES = re.compile(
    r"\b(if\s+confirmed|affected\s+hosts?|all\b|any\b)\b",
    re.IGNORECASE,
)


def _allowed_actions(structured: dict[str, Any]) -> list[str]:
    actions = structured.get("actions") or []
    return [a.get("action") for a in actions if isinstance(a, dict) and a.get("action")]


def _target_is_specific(target: str, episode: Episode) -> bool:
    """Target must contain at least one entity id or artifact value (and not be vague)."""
    if not target or not target.strip():
        return False
    t = target.strip()
    if VAGUE_TARGET_PHRASES.search(t):
        return False
    # Reject generic phrases that don't reference concrete entity/artifact
    if re.search(r"episode\s+entities|entities\s+in\s+episode", t, re.IGNORECASE):
        return False
    entities = set(episode.entities or [])
    artifact_values = set()
    for a in episode.artifacts or []:
        if isinstance(a, dict) and a.get("value") is not None:
            artifact_values.add(str(a.get("value", "")).strip())
    for e in entities:
        if e and e in t:
            return True
    for v in artifact_values:
        if v and v in t:
            return True
    return False


def validate_response_advisor_policy(
    output: AgentOutput,
    episode: Episode,
) -> dict[str, Any]:
    """
    For response_advisor only: action allowlist, target specific (no vague phrases; must reference entity or artifact), duration null -> guardrails + rollback required.
    Return: { valid: bool, errors: list[str] }
    """
    if output.agent_id != "response_advisor":
        return {"valid": True, "errors": []}
    errors: list[str] = []
    structured = output.structured or {}
    actions = structured.get("actions") or []
    for i, a in enumerate(actions):
        if not isinstance(a, dict):
            errors.append(f"actions[{i}]: not a dict")
            continue
        action = (a.get("action") or "").strip().lower()
        if action not in ACTION_ALLOWLIST:
            errors.append(f"actions[{i}].action '{action}' not in allowlist {sorted(ACTION_ALLOWLIST)}")
        target = a.get("target") or ""
        if not _target_is_specific(target, episode):
            errors.append(f"actions[{i}].target '{target}' is vague or does not reference episode entity/artifact")
        duration = a.get("duration_minutes")
        if duration is None:
            guardrails = (a.get("guardrails") or "").strip()
            rollback = a.get("rollback_conditions")
            if not guardrails or len(guardrails) < 10:
                errors.append(f"actions[{i}]: duration_minutes is null but guardrails missing or too short")
            if not rollback or not isinstance(rollback, list) or len(rollback) == 0:
                errors.append(f"actions[{i}]: duration_minutes is null but rollback_conditions missing or empty")
    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }
