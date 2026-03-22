"""Wrapper around src.validators.citations for EvidenceOps."""
from __future__ import annotations

from typing import Any

from src.contracts.agent_output import AgentOutput
from src.contracts.evidence import EvidenceSet
from src.validators.citations import validate_citations as _validate_citations


def validate_agent_citations(
    output: AgentOutput,
    evidence_set: EvidenceSet,
    min_citations: int = 3,
) -> dict[str, Any]:
    return _validate_citations(output, evidence_set, min_citations=min_citations)
