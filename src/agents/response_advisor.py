"""
Response advisor agent: evidence-only. Outputs AgentOutput with ResponsePlan-shaped structured:
actions (action, target, duration_minutes, guardrails, rollback_conditions), expected_impact.
Conservative if confidence < 0.6: prefer collect_more_data / watchlist. >= 3 citations.
"""
from __future__ import annotations

from typing import Any, Optional

from src.contracts.agent_output import AgentOutput
from src.contracts.episode import Episode
from src.contracts.evidence import EvidenceSet
from src.agents.common import PROMPT_VERSION, format_evidence_context


def _specific_targets(episode: Episode) -> str:
    """Build a specific target string from episode.entities and artifact values (no vague phrases)."""
    parts = list(episode.entities or [])
    for a in episode.artifacts or []:
        if isinstance(a, dict) and a.get("value"):
            parts.append(str(a["value"]).strip())
    return ",".join(parts) if parts else "episode-entities"


def run_response_advisor(
    episode: Episode,
    evidence_set: EvidenceSet,
    trust_signals: Optional[dict[str, Any]] = None,
    repair_hint: Optional[str] = None,
) -> AgentOutput:
    """Evidence-only response plan. When repair_hint is set, use only specific targets (entities/artifacts). Conservative if confidence < 0.6."""
    context = format_evidence_context(evidence_set)
    items = evidence_set.items
    top = items[: max(3, min(5, len(items)))] if items else []
    citation_ids = [e.evidence_id for e in top]
    while len(citation_ids) < 3 and len(citation_ids) < len(items):
        citation_ids.append(items[len(citation_ids)].evidence_id)
    citation_ids = citation_ids[: max(3, len(citation_ids))]

    # Decide confidence from evidence and tags
    tags = set(episode.sequence_tags or [])
    confidence = 0.55 if len(items) >= 3 else 0.4
    if "exfil" in tags or "lateral" in tags:
        confidence = min(0.75, confidence + 0.1)
    conservative = confidence < 0.6

    specific_target = _specific_targets(episode)
    actions: list[dict[str, Any]] = []
    if conservative:
        actions.append({
            "action": "collect_more_data",
            "target": specific_target,
            "duration_minutes": None,
            "guardrails": "Do not block or isolate until triage confirms; use evidence to scope collection.",
            "rollback_conditions": ["If triage downgrades to noise, cancel collection."],
        })
        actions.append({
            "action": "watchlist",
            "target": specific_target,
            "duration_minutes": 60,
            "guardrails": "Monitor only; no blocking.",
            "rollback_conditions": ["Remove from watchlist when hypothesis is refuted or contained."],
        })
    else:
        actions.append({
            "action": "watchlist",
            "target": specific_target,
            "duration_minutes": 120,
            "guardrails": "Escalate to isolate if lateral or exfil confirmed.",
            "rollback_conditions": ["Revert if false positive reported or evidence does not support."],
        })
        if "exfil" in tags or "lateral" in tags:
            # Use specific target (never vague "affected hosts if confirmed") so policy guardrails pass
            actions.append({
                "action": "isolate",
                "target": specific_target,
                "duration_minutes": None,
                "guardrails": "Only after triage critical and hunt findings support; apply to listed entities/hosts only.",
                "rollback_conditions": ["Restore network when incident closed or false positive."],
            })

    expected_impact = "Minimal impact (watchlist/collect_more_data only)." if conservative else "Possible isolation of hosts if evidence supports; otherwise watchlist and monitoring."

    structured: dict[str, Any] = {
        "actions": actions,
        "expected_impact": expected_impact,
    }

    next_required: list[str] = []
    if conservative:
        next_required = ["Higher-confidence triage or hunt results before block/isolate"]

    return AgentOutput(
        agent_id="response_advisor",
        episode_id=episode.episode_id,
        run_id=episode.run_id,
        summary=f"Response: {'Conservative (collect_more_data, watchlist).' if conservative else 'Watchlist; isolate if supported.'} {expected_impact}",
        confidence=confidence,
        citations=citation_ids,
        assumptions=["Evidence and triage/hunt outputs are the only basis for actions.", "No production change without validation."],
        next_required_data=next_required,
        structured=structured,
    )
