"""Build EvidenceOps decision bundle JSON and optional legacy audit hashes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.contracts.episode import Episode
from src.contracts.evidence import EvidenceSet
from src.layer_c.schemas.decision_bundle import CaseState, DecisionBundleEvidenceOps
from src.memory.decision_bundle import save_decision_bundle


def _hash_dict(d: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:24]


def build_and_save_evidenceops_bundle(
    case_state: CaseState,
    episode: Episode,
    evidence_set: EvidenceSet,
    repo_root: Path,
    *,
    save_legacy_audit: bool = True,
) -> tuple[Path, DecisionBundleEvidenceOps]:
    """
    Write outputs/evidenceops/decision_bundle_<episode_id>.json (full payload).
    Optionally write legacy outputs/audit/decision_bundle_<episode_id>.json via save_decision_bundle.
    """
    ep_id = episode.episode_id
    out_dir = repo_root / "outputs" / "evidenceops"
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_hash = _hash_dict(episode.model_dump())
    ev_raw = json.loads(evidence_set.model_dump_json())
    evidence_hash = _hash_dict(ev_raw)
    outputs_raw = {k: v.model_dump() for k, v in case_state.by_agent_id.items()}
    outputs_hash = _hash_dict(outputs_raw)

    bundle = DecisionBundleEvidenceOps(
        episode_id=ep_id,
        run_id=episode.run_id,
        case_state=case_state,
        audit_hashes={
            "episode_hash": episode_hash,
            "evidence_hash": evidence_hash,
            "outputs_hash": outputs_hash,
        },
        orchestrator_version=case_state.orchestrator_version,
    )

    path = out_dir / f"decision_bundle_{ep_id}.json"
    path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")

    if save_legacy_audit:
        from src.config import get_config

        cfg = get_config()
        pv = getattr(cfg, "PROMPT_VERSION", "v0.1")
        save_decision_bundle(
            episode_id=ep_id,
            episode_hash=episode_hash,
            evidence_hash=evidence_hash,
            outputs_hash=outputs_hash,
            prompt_version=pv,
            model="evidenceops",
            latency_summary={"orchestrator": case_state.orchestrator_version},
        )

    return path, bundle
