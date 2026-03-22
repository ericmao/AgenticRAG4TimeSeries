"""
Validate merged CaseState: citations per agent; policy on response_advisor only.
"""
from __future__ import annotations

from typing import Any

from src.contracts.episode import Episode
from src.contracts.evidence import EvidenceSet
from src.layer_c.schemas.decision_bundle import CaseState
from src.layer_c.validators.citation_validator import validate_agent_citations
from src.layer_c.validators.policy_guardrails import validate_response_policy


def validate_case_state(
    state: CaseState,
    episode: Episode,
    evidence_set: EvidenceSet,
    min_citations: int = 3,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for agent_id, out in state.by_agent_id.items():
        cr = validate_agent_citations(out, evidence_set, min_citations=min_citations)
        if not cr["valid"]:
            for e in cr.get("errors") or []:
                errors.append(f"{agent_id}: {e}")
        if agent_id == "response_advisor":
            pr = validate_response_policy(out, episode)
            if not pr["valid"]:
                for e in pr.get("errors") or []:
                    errors.append(f"response_advisor policy: {e}")
    return len(errors) == 0, errors


def validate_decision_bundle_schema(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    """Lightweight structural check for serialized EvidenceOps bundle."""
    errs: list[str] = []
    if payload.get("schema_version") != "evidenceops.v1":
        errs.append("schema_version must be evidenceops.v1")
    if not payload.get("episode_id"):
        errs.append("missing episode_id")
    if not payload.get("case_state"):
        errs.append("missing case_state")
    return len(errs) == 0, errs
