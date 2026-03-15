"""
Placeholder heuristic scorer. Outputs Hypothesis v1: ttp_candidates, likelihood, trust_delta, risk_score, recommendation.
If a trained model exists, load it; else use this heuristic.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Optional

from contracts import Hypothesis


def get_device() -> str:
    """MODEL_DEVICE=auto|cpu|cuda. Prefer cuda if available and auto."""
    mode = os.environ.get("MODEL_DEVICE", "auto").strip().lower()
    if mode == "cpu":
        return "cpu"
    if mode == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"
    # auto
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def heuristic_scorer(
    features: dict[str, Any],
    job_id: str | None,
    tenant_id: str,
    endpoint_id: str,
    *,
    window: Optional[dict[str, int]] = None,
    entity_id: Optional[str] = None,
) -> tuple[Hypothesis, dict[str, Any]]:
    """
    MVP heuristic: no trained model. Output Hypothesis v1 + metrics_extra.
    Returns (hypothesis, metrics_extra).
    """
    ec = features.get("event_count", 0)
    ud = features.get("unique_destinations", 0)
    burst = features.get("burstiness", 0.0)
    ttp_candidates: list[dict[str, Any]] = []
    if ud >= 2:
        ttp_candidates.append({"technique_id": "T1021", "technique_name": "Remote Services", "confidence": 0.6})
    if burst >= 10:
        ttp_candidates.append({"technique_id": "T1046", "technique_name": "Network Service Discovery", "confidence": 0.5})
    if ec > 50:
        ttp_candidates.append({"technique_id": "T1078", "technique_name": "Valid Accounts", "confidence": 0.5})
    likelihood = min(1.0, 0.2 + 0.02 * ud + 0.01 * burst)
    risk_score = min(1.0, 0.1 + 0.015 * ec + 0.02 * ud)
    trust_delta = -0.1 * len(ttp_candidates)
    if risk_score >= 0.6:
        recommendation = "Escalate to triage; consider watchlist or collect_more_data."
    else:
        recommendation = "Monitor; no immediate action."
    hypothesis_id = f"hyp-{job_id or endpoint_id}-{tenant_id}"[:64]
    text = f"Heuristic: {len(ttp_candidates)} TTP candidate(s); risk_score={risk_score:.2f}. {recommendation}"
    produced_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    hypothesis = Hypothesis(
        hypothesis_id=hypothesis_id,
        entity_id=entity_id or endpoint_id,
        window=window,
        ttp_candidates=ttp_candidates,
        likelihood=likelihood,
        trust_delta=trust_delta,
        risk_score=risk_score,
        recommendation=recommendation,
        produced_at=produced_at,
        model_version="heuristic-v1",
        text=text,
        suspected_intrusion_set="Unknown" if ttp_candidates else None,
        suspected_tactics=[t.get("technique_id", t.get("id", "")) for t in ttp_candidates],
        constraints=["scope: inference MVP", "no production impact"],
    )
    metrics_extra = {
        "ttp_candidates": ttp_candidates,
        "likelihood": likelihood,
        "trust_delta": trust_delta,
        "risk_score": risk_score,
        "recommendation": recommendation,
        "device": get_device(),
    }
    return hypothesis, metrics_extra
