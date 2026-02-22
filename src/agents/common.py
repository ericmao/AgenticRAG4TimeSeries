"""
Shared evidence formatting and base system prompt for C2 agents. Evidence-only: no external facts.
"""
from __future__ import annotations

from src.contracts.evidence import EvidenceSet

PROMPT_VERSION = "v0.3"

BASE_SYSTEM_PROMPT = (
    "Use only the provided evidence. Every key claim must cite evidence_id. "
    "Do not introduce facts outside the evidence. "
    "Do not introduce new entities or artifacts; reuse episode entities and artifacts verbatim. "
    "When referring to a host, user, IP, or file, copy it from Episode artifacts or entities. "
    "If evidence is insufficient, set lower confidence and list next_required_data; still cite what was used."
)


def format_evidence_context(evidence_set: EvidenceSet) -> list[str]:
    """Return compact text list: '[evidence_id] title :: first_200_chars(body)'."""
    lines: list[str] = []
    for item in evidence_set.items:
        body_preview = (item.body or "")[:200]
        if len((item.body or "")) > 200:
            body_preview += "..."
        lines.append(f"[{item.evidence_id}] {item.title} :: {body_preview}")
    return lines
