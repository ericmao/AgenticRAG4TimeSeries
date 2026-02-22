"""
C3 Writeback: Pydantic models for OpenCTI writeback patch. Each item carries provenance.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Provenance(BaseModel):
    """Provenance for writeback items: evidence_ids, generated_at_ms, source."""

    evidence_ids: list[str] = Field(default_factory=list)
    generated_at_ms: int = 0
    source: str = "pipeline"


class WritebackPatch(BaseModel):
    """Patch to apply to OpenCTI: sightings, relationships, notes; mode dry_run | review | auto. Each item carries provenance."""

    episode_id: str
    run_id: str
    mode: Literal["dry_run", "review", "auto"] = "dry_run"
    sightings: list[dict[str, Any]] = Field(default_factory=list)
    relationships: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[dict[str, Any]] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)
