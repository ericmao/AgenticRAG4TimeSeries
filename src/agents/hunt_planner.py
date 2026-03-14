"""
Hunt planner agent: evidence-only. Outputs AgentOutput with HuntPlan-shaped structured:
queries: {wazuh_query, osquery, kql, suricata_idea}, pivots: {entities, artifacts}, expected_findings.
Queries include episode artifacts when available. >= 3 citations.
"""
from __future__ import annotations

from typing import Any, Optional

from src.contracts.agent_output import AgentOutput
from src.contracts.episode import Episode
from src.contracts.evidence import EvidenceSet
from src.agents.common import PROMPT_VERSION, format_evidence_context


def run_hunt_planner(
    episode: Episode,
    evidence_set: EvidenceSet,
    trust_signals: Optional[dict[str, Any]] = None,
    repair_hint: Optional[str] = None,
) -> AgentOutput:
    """Evidence-only hunt plan. structured: queries (by type), pivots (entities, artifacts), expected_findings. >= 3 citations."""
    context = format_evidence_context(evidence_set)
    items = evidence_set.items
    top = items[: max(3, min(5, len(items)))] if items else []
    citation_ids = [e.evidence_id for e in top]
    while len(citation_ids) < 3 and len(citation_ids) < len(items):
        citation_ids.append(items[len(citation_ids)].evidence_id)
    citation_ids = citation_ids[: max(3, len(citation_ids))]

    # Build queries from episode artifacts
    wazuh_query: list[str] = []
    osquery: list[str] = []
    kql: list[str] = []
    suricata_idea: list[str] = []
    for a in episode.artifacts or []:
        if not isinstance(a, dict):
            continue
        t = (a.get("type") or "").lower()
        v = a.get("value")
        if v is None:
            continue
        val = str(v).strip()
        if t in ("ip", "domain", "address"):
            wazuh_query.append(f'host.ip:{val}')
            kql.append(f'HostIp == "{val}"')
        elif t in ("file", "path"):
            osquery.append(f"SELECT * FROM file WHERE path = '{val}'")
        elif t in ("hash", "sha256", "md5"):
            wazuh_query.append(f'syscheck.summary.hash:{val}')

    # Pivots from episode
    entities = list(episode.entities or [])
    artifacts = [str(a.get("value", "")) for a in (episode.artifacts or []) if isinstance(a, dict) and a.get("value")]

    expected_findings: list[str] = []
    for e in top:
        if "lateral" in (e.body or "").lower():
            expected_findings.append("Signs of lateral movement (logons, process creation)")
        if "exfil" in (e.body or "").lower():
            expected_findings.append("Data exfiltration or unusual outbound traffic")
    if not expected_findings:
        expected_findings = ["Relevant log and process activity for episode entities and artifacts (see evidence)."]

    structured: dict[str, Any] = {
        "queries": {
            "wazuh_query": wazuh_query,
            "osquery": osquery,
            "kql": kql,
            "suricata_idea": suricata_idea,
        },
        "pivots": {"entities": entities, "artifacts": artifacts},
        "expected_findings": expected_findings,
    }

    confidence = 0.65 if (entities or artifacts) and len(items) >= 2 else 0.45
    next_required: list[str] = []
    if not (wazuh_query or osquery or kql):
        next_required.append("Episode artifacts (ip/domain/file/hash) to generate concrete queries")

    return AgentOutput(
        agent_id="hunt_planner",
        episode_id=episode.episode_id,
        run_id=episode.run_id,
        summary=f"Hunt plan: pivots entities={len(entities)} artifacts={len(artifacts)}; queries built from episode. Expected: {'; '.join(expected_findings[:2])}.",
        confidence=confidence,
        citations=citation_ids,
        assumptions=["Episode artifacts and entities are accurate.", "Evidence describes applicable hunt logic."],
        next_required_data=next_required,
        structured=structured,
    )
