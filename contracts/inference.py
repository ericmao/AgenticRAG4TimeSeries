"""
Layer B / Control Plane aligned contracts (MVP + v1).
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
    event_id: Optional[str] = None  # v1 optional


class InferenceRequest(BaseModel):
    """Request for Layer B inference (from Control Plane or job config). v1 aligned."""
    request_id: Optional[str] = None  # v1; fallback to job_id
    job_id: Optional[str] = None
    tenant_id: str = "tenant-default"
    endpoint_id: str
    t0_ms: Optional[int] = None  # legacy
    t1_ms: Optional[int] = None
    window: Optional[dict[str, int]] = None  # v1: { start, end } in ms
    priority: Optional[str] = None  # high | normal | low
    event_refs: Optional[list[str]] = None

    def get_start_end_ms(self) -> tuple[int, int]:
        """Return (start_ms, end_ms) from window or t0_ms/t1_ms."""
        if self.window:
            return (self.window.get("start", 0), self.window.get("end", 0))
        return (self.t0_ms or 0, self.t1_ms or 0)


class TTPCandidate(BaseModel):
    """Single TTP candidate in Hypothesis v1."""
    technique_id: Optional[str] = None
    id: Optional[str] = None  # alias
    technique_name: Optional[str] = None
    name: Optional[str] = None
    tactic: Optional[str] = None
    confidence: float = 0.0


class Hypothesis(BaseModel):
    """Hypothesis output v1 (Layer B). Aligns with Control Plane."""
    hypothesis_id: str = ""
    entity_id: Optional[str] = None  # v1
    window: Optional[dict[str, int]] = None  # v1: { start, end }
    ttp_candidates: list[dict[str, Any]] = Field(default_factory=list)  # v1
    likelihood: Optional[float] = None
    trust_delta: Optional[float] = None
    risk_score: Optional[float] = None
    recommendation: Optional[str] = None
    produced_at: Optional[str] = None  # ISO string
    model_version: Optional[str] = None
    text: str = ""
    suspected_intrusion_set: Optional[str] = None
    suspected_tactics: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


class InferenceResult(BaseModel):
    """Payload POSTed to Control Plane POST /api/v1/inference/results. v1 aligned."""
    request_id: Optional[str] = None
    job_id: Optional[str] = None
    tenant_id: str = "tenant-default"
    endpoint_id: str = ""
    hypothesis: Hypothesis
    produced_at: Optional[str] = None
    model_version: Optional[str] = None
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="e.g. fetch_time_ms, feature_time_ms, inference_time_ms, post_time_ms",
    )
    status: Literal["success", "failed"] = "success"
    error_message: Optional[str] = None
