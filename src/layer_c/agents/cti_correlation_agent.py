"""
CTI Correlation Agent — deterministic, evidence-grounded only.
Correlates evidence items that look like IOC/CTI (source or body keywords) without external lookups.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from src.contracts.agent_output import AgentOutput
from src.contracts.episode import Episode
from src.contracts.evidence import EvidenceSet

_IOC_HINT = re.compile(
    r"\b(ioc|indicator|threat\s*actor|malware|cve-\d+|ttps?|mitre|stix|hash|sha256)\b",
    re.IGNORECASE,
)


def _pick_citations(evidence_set: EvidenceSet, minimum: int = 3) -> list[str]:
    items = evidence_set.items
    if not items:
        return []
    # Prefer items that look CTI-related
    scored: list[tuple[int, str]] = []
    for it in items:
        score = 0
        if (it.source or "").lower() in ("opencti", "cti", "misp"):
            score += 2
        text = f"{it.title or ''} {it.body or ''}"
        if _IOC_HINT.search(text):
            score += 1
        scored.append((score, it.evidence_id))
    scored.sort(key=lambda x: (-x[0], x[1]))
    ordered_ids = [x[1] for x in scored]
    out: list[str] = []
    for eid in ordered_ids:
        if eid not in out:
            out.append(eid)
        if len(out) >= max(minimum, min(5, len(items))):
            break
    while len(out) < minimum and len(out) < len(items):
        for it in items:
            if it.evidence_id not in out:
                out.append(it.evidence_id)
                break
    return out[: max(minimum, len(out))]


def run_cti_correlation(
    episode: Episode,
    evidence_set: EvidenceSet,
    trust_signals: Optional[dict[str, Any]] = None,
    repair_hint: Optional[str] = None,
) -> AgentOutput:
    """List correlated evidence clusters from EvidenceSet only."""
    items = evidence_set.items
    citation_ids = _pick_citations(evidence_set, minimum=3)
    clusters: list[dict[str, Any]] = []
    cti_like = []
    for it in items:
        text = f"{it.title or ''} {it.body or ''}"
        flag = (it.source or "").lower() in ("opencti", "cti", "misp") or bool(_IOC_HINT.search(text))
        if flag:
            cti_like.append(it.evidence_id)
    clusters.append(
        {
            "name": "cti_like_evidence",
            "evidence_ids": sorted(set(cti_like))[:20],
        }
    )

    hyp = (trust_signals or {}).get("hypothesis") if trust_signals else None
    if isinstance(hyp, dict) and hyp.get("suspected_intrusion_set"):
        clusters.append(
            {
                "name": "hypothesis_intrusion_set",
                "value": hyp.get("suspected_intrusion_set"),
                "note": "Hypothesis metadata only; not external CTI.",
            }
        )

    summary = f"CTI-style correlation over {len(items)} evidence items; {len(cti_like)} flagged as CTI-like in-set."
    if repair_hint:
        summary += " (repair applied)"

    return AgentOutput(
        agent_id="cti_correlation",
        episode_id=episode.episode_id,
        run_id=episode.run_id,
        summary=summary,
        confidence=0.7 if len(citation_ids) >= 3 else 0.4,
        citations=citation_ids,
        assumptions=[],
        next_required_data=[],
        structured={
            "clusters": clusters,
        },
    )
