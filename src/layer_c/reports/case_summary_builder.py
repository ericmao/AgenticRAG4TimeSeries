"""Build CaseSummary JSON from CaseState."""
from __future__ import annotations

import json
from pathlib import Path

from src.layer_c.schemas.decision_bundle import CaseState, CaseSummary


def build_case_summary(case_state: CaseState, repo_root: Path) -> tuple[CaseSummary, Path]:
    tri = case_state.by_agent_id.get("triage")
    tl = None
    if tri and isinstance(tri.structured, dict):
        tl = tri.structured.get("triage_level")

    findings: list[str] = []
    for aid in ("entity_investigation", "cti_correlation"):
        o = case_state.by_agent_id.get(aid)
        if o:
            findings.append(f"{aid}: {o.summary[:200]}")

    total_cites = sum(len(o.citations) for o in case_state.by_agent_id.values())

    summary = CaseSummary(
        episode_id=case_state.episode_id,
        run_id=case_state.run_id,
        triage_level=tl,
        agents_executed=list(case_state.by_agent_id.keys()),
        routing_summary=json.dumps(case_state.routing, ensure_ascii=False),
        key_findings=findings,
        evidence_citation_count=total_cites,
    )

    out_dir = repo_root / "outputs" / "evidenceops"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"case_summary_{case_state.episode_id}.json"
    path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    return summary, path
