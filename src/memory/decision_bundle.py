"""
Save decision bundle JSON: episode_hash, evidence_hash, outputs_hash, prompt_version, model, latency summary.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _hash_dict(d: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:24]


def save_decision_bundle(
    episode_id: str,
    episode_hash: str,
    evidence_hash: str,
    outputs_hash: str,
    prompt_version: str = "v0.1",
    model: Optional[str] = None,
    latency_summary: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write outputs/audit/decision_bundle_<episode_id>.json with hashes and metadata."""
    root = _repo_root()
    audit_dir = root / "outputs" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    path = audit_dir / f"decision_bundle_{episode_id}.json"
    payload = {
        "episode_id": episode_id,
        "episode_hash": episode_hash,
        "evidence_hash": evidence_hash,
        "outputs_hash": outputs_hash,
        "prompt_version": prompt_version,
        "model": model or "local",
        "latency_summary": latency_summary or {},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
