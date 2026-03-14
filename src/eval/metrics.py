"""
Evaluation metrics: EGR (Evidence Grounding Rate), UCR_proxy (Unsupported Claim Rate), latency.
UCR uses controlled vocabulary and mention extraction (entities, artifacts, evidence_ids, quoted/IP/path/hash).
"""
from __future__ import annotations

import re
from typing import Any, Dict, Set

from src.contracts.agent_output import AgentOutput
from src.contracts.episode import Episode
from src.contracts.evidence import EvidenceSet

MIN_CITATIONS = 3

# Controlled vocabulary: ignore these (treat as supported). Min 3 chars; short words ignored in token allowlist.
UCR_GENERIC_ALLOWLIST = [
    "lateral movement", "exfiltration", "malware", "suspicious", "anomaly", "triage", "indicator",
    "process creation", "outbound traffic", "credential", "scan", "pivot", "investigate",
    "watchlist", "isolate", "block", "containment", "escalate", "rollback", "hosts", "findings",
    "movement", "hunt", "pivots", "queries", "logons", "evidence", "response", "monitor", "scope",
    "critical", "possible", "otherwise", "listed", "impact", "support", "revert", "reported",
    "confirmed", "apply", "affected", "validation", "recommendations",
    "isolation", "monitoring", "minimal", "duration", "guardrails", "conditions", "expected",
    "actions", "target", "allowlist", "collect_more_data", "wazuh", "osquery", "kql", "suricata",
]
MIN_MENTION_LEN = 3


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


def _tokenize_min_len(text: str, min_len: int = MIN_MENTION_LEN) -> set[str]:
    """Lowercase alphanumeric tokens, min length min_len."""
    if not text:
        return set()
    tokens = set(re.findall(r"[a-zA-Z0-9]+", (text or "").lower()))
    return {t for t in tokens if len(t) >= min_len}


def _flatten_struct(obj: Any) -> str:
    """Flatten dict/list to space-separated string."""
    if isinstance(obj, dict):
        return " ".join(_flatten_struct(v) for v in obj.values())
    if isinstance(obj, list):
        return " ".join(_flatten_struct(v) for v in obj)
    return str(obj) if obj is not None else ""


def _build_supported_set(episode: Episode, evidence_set: EvidenceSet) -> Set[str]:
    """Supported = exact entities, artifact values, evidence_ids, evidence title/body tokens (>=3), generic allowlist tokens."""
    supported: Set[str] = set()
    for e in episode.entities or []:
        if e and len(e) >= MIN_MENTION_LEN:
            supported.add(e)
            supported.add(e.lower())
    for a in episode.artifacts or []:
        if isinstance(a, dict) and a.get("value"):
            v = str(a["value"]).strip()
            if len(v) >= MIN_MENTION_LEN:
                supported.add(v)
                supported.add(v.lower())
    for item in evidence_set.items:
        supported.add(item.evidence_id)
        evidence_text = ((item.title or "") + " " + (item.body or "")).lower()
        supported.update(_tokenize_min_len(evidence_text))
    for phrase in UCR_GENERIC_ALLOWLIST:
        supported.update(_tokenize_min_len(phrase))
    return supported


def _extract_mentions(text: str, episode: Episode, evidence_set: EvidenceSet) -> Set[str]:
    """
    Extract mentions: exact episode entities, artifact values, evidence_id refs, quoted strings, IPs/paths/hashes.
    Do not treat every token as a mention.
    """
    text_lower = (text or "").lower()
    mentions: Set[str] = set()

    for e in episode.entities or []:
        if e and len(e) >= MIN_MENTION_LEN and (e in text or e.lower() in text_lower):
            mentions.add(e)
            mentions.add(e.lower())
    for a in episode.artifacts or []:
        if isinstance(a, dict) and a.get("value"):
            v = str(a["value"]).strip()
            if len(v) >= MIN_MENTION_LEN and (v in text or v.lower() in text_lower):
                mentions.add(v)
                mentions.add(v.lower())
    for item in evidence_set.items:
        if item.evidence_id in text:
            mentions.add(item.evidence_id)

    for m in re.findall(r'"([^"]{3,})"', text):
        mentions.add(m)
        mentions.add(m.lower())
    for m in re.findall(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", text):
        mentions.add(m)
    for m in re.findall(r"\b[a-fA-F0-9]{32,64}\b", text):
        mentions.add(m)
    for m in re.findall(r"[/\\][^\s\"']+", text):
        if len(m) >= MIN_MENTION_LEN:
            mentions.add(m)

    return mentions


def ucr_proxy(
    episode: Episode,
    evidence_set: EvidenceSet,
    agent_outputs: Dict[str, AgentOutput],
) -> float:
    """
    Unsupported Claim Rate proxy: only extracted mentions (entities, artifacts, evidence_ids, quoted, IP/path/hash).
    A mention is supported if in EvidenceItem.title/body, episode entities/artifacts, or generic allowlist.
    ucr_proxy = unsupported_mentions / max(1, total_mentions).
    """
    supported = _build_supported_set(episode, evidence_set)
    total_mentions = 0
    unsupported_mentions = 0
    for out in agent_outputs.values():
        text = (out.summary or "") + " " + _flatten_struct(out.structured)
        mentions = _extract_mentions(text, episode, evidence_set)
        for m in mentions:
            if len(m) < MIN_MENTION_LEN:
                continue
            total_mentions += 1
            if m not in supported and m.lower() not in supported:
                unsupported_mentions += 1
    return unsupported_mentions / max(1, total_mentions)
