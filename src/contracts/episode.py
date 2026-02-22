"""
Layer C: Episode and Hypothesis contracts (Pydantic v2).
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class Episode(BaseModel):
    """Single episode of analysis (time-bounded, entity-scoped)."""

    episode_id: str
    run_id: str
    t0_ms: int
    t1_ms: int
    entities: list[str] = Field(default_factory=list, description="Entity identifiers in scope")
    artifacts: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Artifacts: each {type, value}",
    )
    sequence_tags: list[str] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Events: each {ts_ms, entity, action, artifact, source, confidence, domain}",
    )


class Hypothesis(BaseModel):
    """Optional hypothesis for hunt/response planning."""

    hypothesis_id: str
    text: str
    suspected_intrusion_set: Optional[str] = None
    suspected_tactics: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
