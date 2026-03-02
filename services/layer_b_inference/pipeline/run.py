"""
Orchestrate: fetch events -> build sequence -> run model -> produce InferenceResult.
"""
from __future__ import annotations

import time
from typing import Any, Callable

from contracts import InferenceRequest, InferenceResult, NormalizedEvent

from ..features import build_features
from ..models import heuristic_scorer


def events_from_episode_events(raw_events: list[dict[str, Any]]) -> list[NormalizedEvent]:
    """Convert Episode-style events to NormalizedEvent list."""
    out = []
    for e in raw_events:
        artifact = e.get("artifact")
        if isinstance(artifact, str):
            artifact = {"host": artifact}
        elif not isinstance(artifact, dict):
            artifact = {}
        out.append(
            NormalizedEvent(
                ts_ms=int(e.get("ts_ms", 0)),
                entity=str(e.get("entity", "")),
                action=str(e.get("action", "")),
                artifact=artifact,
                source=str(e.get("source", "logon")),
                confidence=float(e.get("confidence", 1.0)),
                domain=str(e.get("domain", "internal")),
            )
        )
    return out


def run_inference(
    request: InferenceRequest,
    events: list[NormalizedEvent],
    fetch_time_ms: float = 0.0,
) -> InferenceResult:
    """
    Build sequence, extract features, run scorer, return InferenceResult.
    events: pre-fetched (caller responsibility); fetch_time_ms from caller.
    """
    t0 = time.perf_counter()
    features = build_features(events)
    feature_time_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    hypothesis, metrics_extra = heuristic_scorer(
        features,
        request.job_id,
        request.tenant_id,
        request.endpoint_id,
    )
    inference_time_ms = (time.perf_counter() - t1) * 1000

    metrics = {
        "fetch_time_ms": round(fetch_time_ms, 3),
        "feature_time_ms": round(feature_time_ms, 3),
        "inference_time_ms": round(inference_time_ms, 3),
        **{k: v for k, v in metrics_extra.items() if k != "device"},
    }
    return InferenceResult(
        job_id=request.job_id,
        tenant_id=request.tenant_id,
        endpoint_id=request.endpoint_id,
        hypothesis=hypothesis,
        metrics=metrics,
        status="success",
    )
