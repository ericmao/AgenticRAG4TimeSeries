"""
Placeholder heuristic scorer. Outputs ttp_candidates, likelihood, trust_delta, risk_score, recommendation.
If a trained model exists, load it; else use this heuristic.
"""
from __future__ import annotations

import os
from typing import Any

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
) -> tuple[Hypothesis, dict[str, Any]]:
    """
    MVP heuristic: no trained model. Output Hypothesis + extra fields (likelihood, risk_score, etc.).
    Returns (hypothesis, metrics_extra).
    """
    ec = features.get("event_count", 0)
    ud = features.get("unique_destinations", 0)
    burst = features.get("burstiness", 0.0)
    # Simple heuristics
    ttp_candidates: list[dict[str, Any]] = []
    if ud >= 2:
        ttp_candidates.append({"id": "T1021", "name": "Remote Services", "confidence": 0.6})
    if burst >= 10:
        ttp_candidates.append({"id": "T1046", "name": "Network Service Discovery", "confidence": 0.5})
    if ec > 50:
        ttp_candidates.append({"id": "T1078", "name": "Valid Accounts", "confidence": 0.5})
    likelihood = min(1.0, 0.2 + 0.02 * ud + 0.01 * burst)
    risk_score = min(1.0, 0.1 + 0.015 * ec + 0.02 * ud)
    trust_delta = -0.1 * len(ttp_candidates)
    if risk_score >= 0.6:
        recommendation = "Escalate to triage; consider watchlist or collect_more_data."
    else:
        recommendation = "Monitor; no immediate action."
    hypothesis_id = f"hyp-{job_id or endpoint_id}-{tenant_id}"[:64]
    text = f"Heuristic: {len(ttp_candidates)} TTP candidate(s); risk_score={risk_score:.2f}. {recommendation}"
    hypothesis = Hypothesis(
        hypothesis_id=hypothesis_id,
        text=text,
        suspected_intrusion_set="Unknown" if ttp_candidates else None,
        suspected_tactics=[t["id"] for t in ttp_candidates],
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
