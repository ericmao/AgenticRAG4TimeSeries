"""
Evaluation metrics: EGR (Evidence Grounding Rate), UCR_proxy (Unsupported Claim Rate), latency.
"""
from __future__ import annotations

import re
from typing import Any, Dict

from src.contracts.agent_output import AgentOutput
from src.contracts.episode import Episode
from src.contracts.evidence import EvidenceSet

MIN_CITATIONS = 3


def _valid_citations_count(output: AgentOutput, evidence_set: EvidenceSet) -> int:
    """Count citation ids that exist in EvidenceSet."""
    valid_ids = {item.evidence_id for item in evidence_set.items}
    return sum(1 for cid in output.citations if cid in valid_ids)


def egr_per_agent(output: AgentOutput, evidence_set: EvidenceSet, required: int = MIN_CITATIONS) -> float:
    """EGR for one agent: min(1.0, valid_citations / required)."""
    valid = _valid_citations_count(output, evidence_set)
    return min(1.0, valid / required) if required else 1.0


def egr_overall(agent_outputs: Dict[str, AgentOutput], evidence_set: EvidenceSet, required: int = MIN_CITATIONS) -> float:
    """Average EGR over triage, hunt_planner, response_advisor."""
    agents = ["triage", "hunt_planner", "response_advisor"]
    vals = [egr_per_agent(agent_outputs[a], evidence_set, required) for a in agents if a in agent_outputs]
    return sum(vals) / len(vals) if vals else 0.0


def _tokenize(text: str) -> set[str]:
    """Lowercase alphanumeric tokens, min length 2."""
    if not text:
        return set()
    tokens = set(re.findall(r"[a-zA-Z0-9]{2,}", (text or "").lower()))
    return tokens


def _flatten_struct(obj: Any) -> str:
    """Flatten dict/list to space-separated string for tokenization."""
    if isinstance(obj, dict):
        return " ".join(_flatten_struct(v) for v in obj.values())
    if isinstance(obj, list):
        return " ".join(_flatten_struct(v) for v in obj)
    return str(obj) if obj is not None else ""


def ucr_proxy(
    episode: Episode,
    evidence_set: EvidenceSet,
    agent_outputs: Dict[str, AgentOutput],
) -> float:
    """
    Unsupported Claim Rate proxy: tokens in summary+structured that are not in
    (episode.entities + episode.artifacts values) and not in any EvidenceItem.body/title.
    ucr_proxy = unsupported_mentions / max(1, total_mentions).
    """
    allowed = set()
    for e in episode.entities or []:
        if e:
            allowed.update(_tokenize(e))
    for a in episode.artifacts or []:
        if isinstance(a, dict) and a.get("value"):
            allowed.update(_tokenize(str(a["value"])))
    evidence_text = " ".join(
        (item.title or "") + " " + (item.body or "") for item in evidence_set.items
    )
    allowed.update(_tokenize(evidence_text))

    total_mentions = 0
    unsupported_mentions = 0
    for out in agent_outputs.values():
        text = (out.summary or "") + " " + _flatten_struct(out.structured)
        tokens = _tokenize(text)
        for t in tokens:
            total_mentions += 1
            if t not in allowed:
                unsupported_mentions += 1
    return unsupported_mentions / max(1, total_mentions)
