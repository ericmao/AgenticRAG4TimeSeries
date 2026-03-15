"""
Tests for Layer B + Layer C integration using sample/CERT data.
- Layer B: job (request + events) -> run_inference -> InferenceResult.
- Build Episode from InferenceResult + events.
- Layer C: retrieve (with hypothesis), analyze, writeback (dry_run).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_layer_b_sample_job_produces_inference_result():
    """Run Layer B with jobs/sample_job.json; assert InferenceResult shape and risk_score."""
    from contracts import InferenceResult
    from services.layer_b_inference.pipeline import events_from_episode_events, run_inference
    from contracts import InferenceRequest

    job_path = REPO_ROOT / "jobs" / "sample_job.json"
    if not job_path.exists():
        pytest.skip("jobs/sample_job.json not found")
    data = json.loads(job_path.read_text(encoding="utf-8"))
    request = InferenceRequest.model_validate(data["request"])
    events = events_from_episode_events(data["events"])
    result = run_inference(request, events, fetch_time_ms=0.0)
    assert result.status == "success"
    assert result.hypothesis is not None
    assert result.hypothesis.risk_score is not None
    assert 0 <= result.hypothesis.risk_score <= 1.0
    assert result.hypothesis.entity_id is not None or result.endpoint_id
    InferenceResult.model_validate(result.model_dump())


def test_build_episode_from_inference_result():
    """Build Episode + Hypothesis from Layer B result and job events."""
    from contracts import InferenceResult
    from services.layer_b_inference.pipeline import events_from_episode_events, run_inference
    from contracts import InferenceRequest
    from src.pipeline.build_episode_from_inference import build_episode_from_inference
    from src.contracts.episode import Episode

    job_path = REPO_ROOT / "jobs" / "sample_job.json"
    if not job_path.exists():
        pytest.skip("jobs/sample_job.json not found")
    data = json.loads(job_path.read_text(encoding="utf-8"))
    request = InferenceRequest.model_validate(data["request"])
    events_raw = data["events"]
    events = events_from_episode_events(events_raw)
    result = run_inference(request, events, fetch_time_ms=0.0)
    episode, lc_hyp = build_episode_from_inference(result, events_raw)
    assert isinstance(episode, Episode)
    assert episode.episode_id
    assert episode.t0_ms <= episode.t1_ms
    assert episode.risk_context is not None
    assert "risk_score" in episode.risk_context
    assert lc_hyp.hypothesis_id
    Episode.model_validate(episode.model_dump())


def test_e2e_episode_from_demo():
    """Load demo episode, build Layer B job, run inference, build Episode; no Layer C calls."""
    demo_path = REPO_ROOT / "tests" / "demo" / "episode_insider_highrisk.json"
    if not demo_path.exists():
        pytest.skip("tests/demo/episode_insider_highrisk.json not found")
    ep_data = json.loads(demo_path.read_text(encoding="utf-8"))
    events = ep_data.get("events", [])[:15]  # small slice
    if not events:
        pytest.skip("no events in demo episode")
    from contracts import InferenceRequest
    from services.layer_b_inference.pipeline import events_from_episode_events, run_inference
    from src.pipeline.build_episode_from_inference import build_episode_from_inference

    request = InferenceRequest(
        request_id=f"req-{ep_data['episode_id']}",
        job_id=f"job-{ep_data['episode_id']}",
        tenant_id="tenant-default",
        endpoint_id=ep_data["entities"][0] if ep_data.get("entities") else "USER0420",
        t0_ms=ep_data["t0_ms"],
        t1_ms=ep_data["t1_ms"],
    )
    norm_events = events_from_episode_events(events)
    result = run_inference(request, norm_events, fetch_time_ms=0.0)
    episode, _ = build_episode_from_inference(result, events, episode_id=ep_data["episode_id"])
    assert episode.episode_id == ep_data["episode_id"]
    assert len(episode.events) == len(events)
    assert episode.risk_context is not None
