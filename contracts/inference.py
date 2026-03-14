"""
Layer B / Control Plane aligned contracts (MVP).
TODO: Replace with package or git submodule when Control Plane contracts are published.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class NormalizedEvent(BaseModel):
    """Single normalized event (aligns with Episode.events[] and event ingestion)."""
    ts_ms: int
    entity: str
    action: str
    artifact: dict[str, Any] = Field(default_factory=dict)  # e.g. {host: "PC001"}
    source: str = "logon"  # logon | device | other
    confidence: float = 1.0
    domain: str = "internal"


class InferenceRequest(BaseModel):
    """Request for Layer B inference (from Control Plane or job config)."""
    job_id: Optional[str] = None
    tenant_id: str
    endpoint_id: str
    t0_ms: int
    t1_ms: int
    event_refs: Optional[list[str]] = None  # optional references to fetch events


class Hypothesis(BaseModel):
    """Hypothesis output (aligns with Control Plane and src.contracts.episode.Hypothesis)."""
    hypothesis_id: str
    text: str
    suspected_intrusion_set: Optional[str] = None
    suspected_tactics: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


class InferenceResult(BaseModel):
    """Payload POSTed to Control Plane POST /api/v1/inference/results."""
    job_id: Optional[str] = None
    tenant_id: str
    endpoint_id: str
    hypothesis: Hypothesis
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="e.g. fetch_time_ms, feature_time_ms, inference_time_ms, post_time_ms",
    )
    status: Literal["success", "failed"] = "success"
    error_message: Optional[str] = None
