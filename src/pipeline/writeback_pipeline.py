"""
C3 Writeback pipeline: load episode, evidence, agent outputs; build patch (dry_run); write writeback + decision bundle; audit_log.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from src.contracts.agent_output import AgentOutput
from src.contracts.episode import Episode
from src.contracts.evidence import EvidenceSet
from src.contracts.writeback import WritebackPatch
from src.memory.decision_bundle import save_decision_bundle
from src.memory.opencti_writeback import apply_patch, build_writeback_patch
from src.utils.audit_logger import audit_log


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _hash_content(data: dict) -> str:
    return __import__("hashlib").sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:24]


def run_writeback(
    episode_path: str | Path,
    mode: str = "dry_run",
) -> tuple[WritebackPatch, Path]:
    """
    Load episode, evidence, agent outputs; build patch (mode); write outputs/writeback/<episode_id>.json,
    outputs/audit/decision_bundle_<episode_id>.json; audit_log stats. Returns (patch, writeback_path).
    """
    root = _repo_root()
    episode_path = Path(episode_path)
    if not episode_path.is_absolute():
        episode_path = root / episode_path
    episode = Episode.model_validate(json.loads(episode_path.read_text(encoding="utf-8")))
    episode_id = episode.episode_id
    run_id = episode.run_id

    evidence_path = root / "outputs" / "evidence" / f"{episode_id}.json"
    if not evidence_path.exists():
        raise FileNotFoundError(f"Evidence not found: {evidence_path}; run retrieve first.")
    evidence_set = EvidenceSet.model_validate(json.loads(evidence_path.read_text(encoding="utf-8")))
    evidence_raw = json.loads(evidence_path.read_text(encoding="utf-8"))

    agents_dir = root / "outputs" / "agents"
    agent_outputs: Dict[str, AgentOutput] = {}
    outputs_raw: Dict[str, dict] = {}
    for agent_id in ("triage", "hunt_planner", "response_advisor"):
        p = agents_dir / f"{episode_id}_{agent_id}.json"
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            agent_outputs[agent_id] = AgentOutput.model_validate(data)
            outputs_raw[agent_id] = data
    if not agent_outputs:
        raise FileNotFoundError(f"No agent outputs under {agents_dir} for episode_id={episode_id}; run analyze first.")

    patch = build_writeback_patch(episode, evidence_set, agent_outputs, mode=mode)
    apply_patch(patch)

    writeback_dir = root / "outputs" / "writeback"
    writeback_dir.mkdir(parents=True, exist_ok=True)
    writeback_path = writeback_dir / f"{episode_id}.json"
    writeback_path.write_text(patch.model_dump_json(indent=2), encoding="utf-8")

    episode_hash = _hash_content(episode.model_dump())
    evidence_hash = _hash_content(evidence_raw)
    outputs_hash = _hash_content(outputs_raw)
    from src.config import get_config
    cfg = get_config()
    prompt_version = getattr(cfg, "PROMPT_VERSION", "v0.1")
    bundle_path = save_decision_bundle(
        episode_id=episode_id,
        episode_hash=episode_hash,
        evidence_hash=evidence_hash,
        outputs_hash=outputs_hash,
        prompt_version=prompt_version,
        model="local",
        latency_summary={},
    )

    audit_log(
        event={
            "action": "writeback",
            "episode_id": episode_id,
            "mode": patch.mode,
            "stats": patch.stats,
            "writeback_path": str(writeback_path.relative_to(root)),
            "decision_bundle_path": str(bundle_path.relative_to(root)),
        },
        run_id=run_id,
        episode_id=episode_id,
        component="pipeline.writeback",
        prompt_version=prompt_version,
    )
    return patch, writeback_path
