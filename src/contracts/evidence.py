"""
Layer C: EvidenceItem and EvidenceSet contracts (Pydantic v2).
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    """Single evidence item; evidence_id is a stable hash of (source, kind, title, body, stix_id, chunk_id)."""

    evidence_id: str
    source: Literal["opencti", "kb", "taxii", "other"]
    kind: Literal["stix_object", "snippet", "relation"]
    title: str
    body: str
    stix_id: Optional[str] = None
    chunk_id: Optional[str] = None
    score: float = Field(ge=0.0, le=1.0)
    ts_ms: Optional[int] = None
    provenance: dict[str, Any] = Field(
        default_factory=dict,
        description="e.g. {retrieved_at_ms, query, url?, content_hash?}",
    )


class EvidenceSet(BaseModel):
    """Collection of evidence items for an episode."""

    episode_id: str
    run_id: str
    items: list[EvidenceItem] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)
    max_items: int = Field(default=50, ge=1, le=1000)
