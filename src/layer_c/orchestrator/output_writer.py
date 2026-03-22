"""Write EvidenceOps multi-agent outputs to outputs/agents/ (compat with writeback + legacy tools)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.contracts.agent_output import AgentOutput


def write_agent_bundle_to_disk(
    episode_id: str,
    rule_id: str,
    bundle: dict[str, AgentOutput],
    repo_root: Path,
) -> None:
    """
    Write <episode_id>_by_rule.json with all agents in bundle.
    Also write flat triage/hunt_planner/response_advisor for backward compatibility.
    """
    out_dir = repo_root / "outputs" / "agents"
    out_dir.mkdir(parents=True, exist_ok=True)
    serial: dict[str, Any] = {rule_id: {k: v.model_dump() for k, v in bundle.items()}}
    by_path = out_dir / f"{episode_id}_by_rule.json"
    by_path.write_text(json.dumps(serial, indent=2, ensure_ascii=False), encoding="utf-8")

    for agent_id in ("triage", "hunt_planner", "response_advisor"):
        if agent_id in bundle:
            p = out_dir / f"{episode_id}_{agent_id}.json"
            p.write_text(bundle[agent_id].model_dump_json(indent=2), encoding="utf-8")
