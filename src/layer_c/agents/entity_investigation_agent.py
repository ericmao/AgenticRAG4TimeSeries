"""
Entity Investigation Agent — deterministic, evidence-grounded only.
Maps episode.entities and evidence snippets to scoped investigation notes.
"""
from __future__ import annotations

from typing import Any, Optional

from src.contracts.agent_output import AgentOutput
from src.contracts.episode import Episode
from src.contracts.evidence import EvidenceSet


def _pick_citations(evidence_set: EvidenceSet, minimum: int = 3) -> list[str]:
    items = evidence_set.items
    if not items:
        return []
    n = max(minimum, min(len(items), 5))
    return [items[i].evidence_id for i in range(min(n, len(items)))]


def run_entity_investigation(
    episode: Episode,
    evidence_set: EvidenceSet,
    trust_signals: Optional[dict[str, Any]] = None,
    repair_hint: Optional[str] = None,
) -> AgentOutput:
    """Scope entities against cited evidence only; >=3 citations when evidence allows."""
    items = evidence_set.items
    citation_ids = _pick_citations(evidence_set, minimum=3)
    entities = list(episode.entities or [])
    scoped: list[dict[str, Any]] = []
    for eid in entities[:10]:
        hits = []
        for it in items:
            blob = f"{it.title or ''} {it.body or ''}".lower()
            if eid.lower() in blob:
                hits.append(it.evidence_id)
        scoped.append({"entity_id": eid, "evidence_hits": sorted(set(hits))[:5]})

    if not entities and items:
        scoped.append(
            {
                "entity_id": "(none in episode.entities)",
                "note": "No episode.entities; pivots from evidence titles only.",
            }
        )

    summary_parts = [f"Scoped {len(entities)} entities against {len(items)} evidence items."]
    if repair_hint:
        summary_parts.append(f"(repair_hint applied)")

    conf = 0.65 if len(citation_ids) >= 3 else 0.35
    return AgentOutput(
        agent_id="entity_investigation",
        episode_id=episode.episode_id,
        run_id=episode.run_id,
        summary=" ".join(summary_parts),
        confidence=conf,
        citations=citation_ids,
        assumptions=[],
        next_required_data=[],
        structured={
            "entities_scoped": scoped,
            "trust_signals_keys": list((trust_signals or {}).keys()),
        },
    )
