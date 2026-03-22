"""
Layer C: AgentOutput base and structured payloads HuntPlan / ResponsePlan (Pydantic v2).
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class AgentOutput(BaseModel):
    """Base output shape for all agents; structured is agent-specific (e.g. HuntPlan, ResponsePlan)."""

    agent_id: Literal[
        "triage",
        "hunt_planner",
        "response_advisor",
        "entity_investigation",
        "cti_correlation",
    ]
    episode_id: str
    run_id: str
    summary: str
    confidence: float = Field(ge=0.0, le=1.0)
    citations: list[str] = Field(default_factory=list, description="evidence_id list")
    assumptions: list[str] = Field(default_factory=list)
    next_required_data: list[str] = Field(default_factory=list)
    structured: dict[str, Any] = Field(default_factory=dict, description="Agent-specific payload")


# ----- Structured payloads for structured field -----


class HuntPlan(BaseModel):
    """Structured payload for hunt_planner agent."""

    queries: list[dict[str, Any]] = Field(
        default_factory=list,
        description="e.g. wazuh_query, osquery, kql, suricata_idea",
    )
    pivots: list[str] = Field(default_factory=list)
    expected_findings: list[str] = Field(default_factory=list)


class ResponsePlan(BaseModel):
    """Structured payload for response_advisor agent."""

    actions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="e.g. allowlist: block | isolate | watchlist | collect_more_data",
    )
    scope: list[str] = Field(default_factory=list)
    duration_minutes: Optional[int] = None
    rollback_conditions: list[str] = Field(default_factory=list)
    expected_impact: Optional[str] = None
