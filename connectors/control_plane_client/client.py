"""
POST InferenceResult to Control Plane (guacamole-ai). Retry with exponential backoff; store failed in outbox.
POST WritebackPatch to Control Plane (guacamole-ai) for Layer C cases writeback.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from contracts import InferenceResult


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _base_url() -> str:
    return os.environ.get("CONTROL_PLANE_BASE_URL", "").strip()


def _token() -> str:
    return os.environ.get("CONTROL_PLANE_TOKEN", "").strip()


def _inference_result_to_guacamole_body(result: InferenceResult) -> dict[str, Any]:
    """
    Convert AgenticRAG InferenceResult to guacamole-ai POST /api/v1/inference/results body.
    Guacamole expects: tenant_id, endpoint_id, hypothesis (entity_id, ttp_candidates with id/name), produced_at, model_version.
    """
    hyp = result.hypothesis
    ttp_list = []
    for t in hyp.ttp_candidates or []:
        if not isinstance(t, dict):
            continue
        ttp_list.append({
            "id": t.get("technique_id") or t.get("id") or "",
            "name": t.get("technique_name") or t.get("name") or "",
            "tactic": t.get("tactic"),
            "confidence": t.get("confidence"),
        })
    body = {
        "tenant_id": result.tenant_id or "tenant-default",
        "endpoint_id": result.endpoint_id or "",
        "hypothesis": {
            "entity_id": hyp.entity_id or result.endpoint_id or "",
            "ttp_candidates": ttp_list,
            "likelihood": hyp.likelihood if hyp.likelihood is not None else 0.0,
            "trust_delta": hyp.trust_delta if hyp.trust_delta is not None else 0.0,
            "risk_score": hyp.risk_score if hyp.risk_score is not None else 0.0,
            "recommendation": hyp.recommendation,
            "produced_at": hyp.produced_at or result.produced_at or "",
        },
        "produced_at": result.produced_at or hyp.produced_at or "",
        "model_version": result.model_version or "",
    }
    return body


def post_result(result: InferenceResult) -> tuple[bool, Optional[str]]:
    """
    POST result to {CONTROL_PLANE_BASE_URL}/api/v1/inference/results (guacamole-ai Layer B).
    Sends guacamole-compatible JSON (id/name for TTP). Returns (success, error_message).
    """
    base = _base_url()
    token = _token()
    if not base or not token:
        return False, "CONTROL_PLANE_BASE_URL or CONTROL_PLANE_TOKEN not set"
    url = f"{base.rstrip('/')}/api/v1/inference/results"
    payload = json.dumps(_inference_result_to_guacamole_body(result))
    # Retry with exponential backoff
    for attempt in range(3):
        try:
            import urllib.request
            req = urllib.request.Request(
                url,
                data=payload.encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                if 200 <= resp.status < 300:
                    return True, None
                return False, f"HTTP {resp.status}"
        except Exception as e:
            if attempt == 2:
                return False, str(e)
            time.sleep(2 ** attempt)
    return False, "max retries exceeded"


def save_outbox(result: InferenceResult, error: str) -> Path:
    """Append failed result to outbox file for later retry."""
    root = _repo_root()
    outbox_dir = root / "outputs" / "layer_b_outbox"
    outbox_dir.mkdir(parents=True, exist_ok=True)
    path = outbox_dir / "failed_results.jsonl"
    line = json.dumps({"result": result.model_dump(), "error": error}) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
    return path


def post_writeback(patch: Any) -> tuple[bool, Optional[str]]:
    """
    POST WritebackPatch to {CONTROL_PLANE_BASE_URL}/api/v1/cases/writeback (guacamole-ai Layer C).
    patch: WritebackPatch or dict with episode_id, run_id, mode, sightings, relationships, notes, stats.
    Returns (success, error_message).
    """
    base = _base_url()
    token = _token()
    if not base or not token:
        return False, "CONTROL_PLANE_BASE_URL or CONTROL_PLANE_TOKEN not set"
    url = f"{base.rstrip('/')}/api/v1/cases/writeback"
    if hasattr(patch, "model_dump"):
        payload = patch.model_dump_json()
    else:
        payload = json.dumps(patch)
    for attempt in range(3):
        try:
            import urllib.request
            req = urllib.request.Request(
                url,
                data=payload.encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                if 200 <= resp.status < 300:
                    return True, None
                return False, f"HTTP {resp.status}"
        except Exception as e:
            if attempt == 2:
                return False, str(e)
            time.sleep(2 ** attempt)
    return False, "max retries exceeded"
