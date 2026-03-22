"""
Deterministic routing from triage_level (EvidenceOps).
"""
from __future__ import annotations

from typing import Any


def triage_level_from_output(structured: dict[str, Any]) -> str:
    """Read triage_level from triage agent structured output."""
    lvl = structured.get("triage_level") if isinstance(structured, dict) else None
    if isinstance(lvl, str) and lvl.strip():
        return lvl.strip().lower()
    return "suspicious"


def plan_route(triage_level: str) -> dict[str, Any]:
    """
    Returns which downstream steps to run.
    - critical / suspicious: full chain including hunt.
    - noise: skip CTI deep correlation and hunt (stubs inserted for compatibility).
    """
    tl = (triage_level or "").lower()
    if tl == "noise":
        return {
            "triage_level": tl,
            "run_entity_investigation": True,
            "run_cti_correlation": False,
            "run_hunt_planner": False,
            "run_response_advisor": True,
            "hunt_stub_reason": "noise_tier_skipped",
        }
    if tl in ("critical", "suspicious"):
        return {
            "triage_level": tl,
            "run_entity_investigation": True,
            "run_cti_correlation": True,
            "run_hunt_planner": True,
            "run_response_advisor": True,
            "hunt_stub_reason": None,
        }
    # unknown tier — conservative full run
    return {
        "triage_level": tl or "unknown",
        "run_entity_investigation": True,
        "run_cti_correlation": True,
        "run_hunt_planner": True,
        "run_response_advisor": True,
        "hunt_stub_reason": None,
    }
