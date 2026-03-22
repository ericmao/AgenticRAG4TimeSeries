"""
EvidenceOps-specific case state and decision bundle shapes (Pydantic v2).
Distinct from outputs/audit/decision_bundle_<id>.json hash-only metadata.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from src.contracts.agent_output import AgentOutput


class RiskContextInput(BaseModel):
    """Optional explicit risk context; defaults from Episode.risk_context."""

    raw: dict[str, Any] = Field(default_factory=dict, description="Merged into orchestrator trust_signals")

    @classmethod
    def from_episode(cls, episode: Any) -> "RiskContextInput":
        rc = getattr(episode, "risk_context", None) or {}
        if not isinstance(rc, dict):
            rc = {}
        return cls(raw=dict(rc))


class HypothesisInput(BaseModel):
    """Optional Layer C hypothesis (hunt/response constraints)."""

    hypothesis: Optional[Any] = None  # Hypothesis model or dict

    def to_trust_signals(self) -> dict[str, Any]:
        if self.hypothesis is None:
            return {}
        if hasattr(self.hypothesis, "model_dump"):
            return {"hypothesis": self.hypothesis.model_dump()}
        if isinstance(self.hypothesis, dict):
            return {"hypothesis": self.hypothesis}
        return {}


class CaseState(BaseModel):
    """Merged multi-agent state after CaseOrchestrator.run."""

    episode_id: str
    run_id: str
    rule_id: str = "default"
    orchestrator_version: str = "0.1.0"
    routing: dict[str, Any] = Field(default_factory=dict)
    by_agent_id: dict[str, AgentOutput] = Field(default_factory=dict)

    def agent_ids(self) -> list[str]:
        return list(self.by_agent_id.keys())


class CaseSummary(BaseModel):
    """Structured summary for UI / audit (JSON-serializable)."""

    episode_id: str
    run_id: str
    triage_level: Optional[str] = None
    agents_executed: list[str] = Field(default_factory=list)
    routing_summary: str = ""
    key_findings: list[str] = Field(default_factory=list)
    evidence_citation_count: int = 0


class DecisionBundleEvidenceOps(BaseModel):
    """EvidenceOps decision bundle payload (extends hash metadata)."""

    schema_version: str = "evidenceops.v1"
    episode_id: str
    run_id: str
    case_state: CaseState
    audit_hashes: dict[str, str] = Field(default_factory=dict, description="episode_hash, evidence_hash, outputs_hash")
    prompt_version: str = "v0.1"
    orchestrator_version: str = "0.1.0"
