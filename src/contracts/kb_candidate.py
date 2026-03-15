"""
KB candidate: document/snippet derived from CERT Episode for potential inclusion in the knowledge base.
Self-evaluation: reliability_score (0-1); when >= threshold, can auto-add to KB.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class KBCandidate(BaseModel):
    """A candidate document derived from an episode for the KB."""

    episode_id: str = Field(description="Source episode ID")
    run_id: str = Field(description="Run ID of the episode")
    sequence_tags: list[str] = Field(
        default_factory=list,
        description="Tags from episode (e.g. logon, lateral, burst)",
    )
    artifacts_summary: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Summary of artifacts by type (e.g. user: [USER1], host: [PC-001, PC-002])",
    )
    body_md: str = Field(description="Markdown body suitable for KB inclusion")
    event_count: int = Field(default=0, description="Number of events in the episode window")
    source: str = Field(default="cert_episode", description="Origin of the candidate")
    reliability_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Self-evaluation score (0-1); >= threshold => auto-add to KB",
    )
    auto_add: bool = Field(
        default=False,
        description="True when reliability_score >= configured threshold",
    )
