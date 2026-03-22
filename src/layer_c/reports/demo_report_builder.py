"""Generate demo report and append EvidenceOps section."""
from __future__ import annotations

from pathlib import Path

from src.layer_c.schemas.decision_bundle import CaseState
from src.pipeline.demo_report import generate_demo_report


def generate_demo_report_with_evidenceops(
    episode_path: str,
    case_state: CaseState | None,
) -> Path:
    """Run existing demo report, then append orchestrator block if case_state provided."""
    report_path = generate_demo_report(episode_path)
    if case_state is None:
        return report_path
    rp = Path(report_path)
    if not rp.is_file():
        return report_path
    block = (
        "\n\n---\n\n## SenseL EvidenceOps\n\n"
        f"- orchestrator_version: `{case_state.orchestrator_version}`\n"
        f"- agents: {', '.join(case_state.by_agent_id.keys())}\n"
        f"- routing: `{case_state.routing}`\n"
    )
    text = rp.read_text(encoding="utf-8") + block
    rp.write_text(text, encoding="utf-8")
    return rp
