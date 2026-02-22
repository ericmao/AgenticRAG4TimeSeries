"""
Triage agent: evidence-only. Outputs AgentOutput with triage_level, why_now, top_evidence, key_risks.
Citations must cover why_now + key_risks; >= 3 evidence_id.
"""
from __future__ import annotations

from typing import Any, Optional

from src.contracts.agent_output import AgentOutput
from src.contracts.episode import Episode
from src.contracts.evidence import EvidenceSet
from src.agents.common import PROMPT_VERSION, format_evidence_context


def run_triage(
    episode: Episode,
    evidence_set: EvidenceSet,
    trust_signals: Optional[dict[str, Any]] = None,
) -> AgentOutput:
    """Evidence-only triage. structured: triage_level, why_now, top_evidence, key_risks. >= 3 citations."""
    context = format_evidence_context(evidence_set)
    # Top evidence by score (already sorted in EvidenceSet); take at least 3 for citations
    items = evidence_set.items
    top = items[: max(3, min(5, len(items)))] if items else []
    citation_ids = [e.evidence_id for e in top]
    while len(citation_ids) < 3 and len(citation_ids) < len(items):
        citation_ids.append(items[len(citation_ids)].evidence_id)
    citation_ids = citation_ids[: max(3, len(citation_ids))]

    # Derive triage_level from sequence_tags + evidence content
    tags = set(episode.sequence_tags or [])
    critical_tags = {"exfil", "lateral", "critical"}
    suspicious_tags = {"suspicious", "login", "anomaly"}
    if tags & critical_tags:
        triage_level = "critical"
    elif tags or suspicious_tags:
        triage_level = "suspicious"
    else:
        triage_level = "noise"

    why_parts = [f"Sequence tags: {', '.join(sorted(episode.sequence_tags or []))}"]
    if trust_signals:
        why_parts.append(f"Trust signals: {trust_signals}")
    why_now = " ".join(why_parts) if why_parts else "No sequence tags; assessment from evidence only."

    # Key risks from evidence titles/bodies (simple extraction)
    key_risks: list[str] = []
    for e in top:
        if "isolate" in (e.body or "").lower() or "block" in (e.title or "").lower():
            key_risks.append("Containment actions (isolate/block) referenced in evidence")
        if "lateral" in (e.body or "").lower() or "exfil" in (e.body or "").lower():
            key_risks.append("Lateral movement or exfil mentioned in evidence")
    if not key_risks:
        key_risks = ["Evidence suggests review of containment and scope (see cited snippets)."]
    key_risks = list(dict.fromkeys(key_risks))[:5]

    confidence = 0.7 if len(items) >= 3 else 0.4
    next_required: list[str] = []
    if len(items) < 3:
        next_required = ["More evidence items to raise triage confidence", "OpenCTI or additional KB docs if available"]

    structured: dict[str, Any] = {
        "triage_level": triage_level,
        "why_now": why_now,
        "top_evidence": citation_ids,
        "key_risks": key_risks,
    }

    return AgentOutput(
        agent_id="triage",
        episode_id=episode.episode_id,
        run_id=episode.run_id,
        summary=f"Triage: {triage_level}. {why_now} Key risks: {'; '.join(key_risks[:2])}.",
        confidence=confidence,
        citations=citation_ids,
        assumptions=["Evidence set is complete for this episode.", "Sequence tags reflect upstream detection logic."],
        next_required_data=next_required,
        structured=structured,
    )
