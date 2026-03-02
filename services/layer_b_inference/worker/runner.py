"""
Pull-mode worker: read job configs from folder (MVP), run inference, POST result.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from contracts import InferenceRequest

from ..pipeline import events_from_episode_events, run_inference

# Optional: control plane client (post result)
try:
    from connectors.control_plane_client import post_result, save_outbox
except ImportError:
    def post_result(_):
        return False, "connectors not available"
    def save_outbox(_, __):
        return Path(".")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_job_config(path: Path) -> tuple[InferenceRequest, list[dict[str, Any]]]:
    """
    Load job config JSON: { request: InferenceRequest dict, events?: list of event dicts }.
    If events missing, return empty events (caller may fetch elsewhere).
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    req = InferenceRequest.model_validate(data["request"])
    events_raw = data.get("events", [])
    return req, events_raw


def run_one_job(
    request: InferenceRequest,
    events_raw: list[dict[str, Any]],
    post_to_control_plane: bool = True,
) -> tuple[Any, float]:
    """
    Run inference for one job. Returns (InferenceResult, fetch_time_ms).
    If post_to_control_plane, POST result and on failure save to outbox.
    """
    t0 = time.perf_counter()
    events = events_from_episode_events(events_raw)
    fetch_time_ms = (time.perf_counter() - t0) * 1000

    result = run_inference(request, events, fetch_time_ms=fetch_time_ms)

    if post_to_control_plane:
        ok, err = post_result(result)
        if not ok:
            save_outbox(result, err or "unknown")
    return result, fetch_time_ms


def run_jobs_from_folder(
    jobs_dir: Path,
    post_to_control_plane: bool = True,
) -> list[tuple[str, Any, float]]:
    """
    Scan jobs_dir for *.json job configs; run each; return list of (job_id_or_path, result, fetch_time_ms).
    """
    if not jobs_dir.is_dir():
        return []
    out = []
    for path in sorted(jobs_dir.glob("*.json")):
        try:
            req, events_raw = load_job_config(path)
            result, fetch_ms = run_one_job(req, events_raw, post_to_control_plane=post_to_control_plane)
            out.append((req.job_id or path.name, result, fetch_ms))
        except Exception as e:
            # Log and continue
            out.append((path.name, None, 0.0))
    return out
