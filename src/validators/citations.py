"""
Citation validity: every AgentOutput.citations[] must exist in EvidenceSet; min citations configurable.
Returns detailed errors: missing_ids, too_few_citations.
"""
from __future__ import annotations

from typing import Any

from src.contracts.agent_output import AgentOutput
from src.contracts.evidence import EvidenceSet


def validate_citations(
    output: AgentOutput,
    evidence_set: EvidenceSet,
    min_citations: int = 3,
) -> dict[str, Any]:
    """
    Check that every output.citations id exists in evidence_set.items[].evidence_id
    and that len(citations) >= min_citations.
    Return: { valid: bool, missing_ids: list[str], too_few_citations: bool, errors: list[str] }
    """
    valid_ids = {item.evidence_id for item in evidence_set.items}
    missing_ids = [cid for cid in output.citations if cid not in valid_ids]
    too_few = len(output.citations) < min_citations
    errors: list[str] = []
    if missing_ids:
        errors.append(f"missing_ids: {missing_ids}")
    if too_few:
        errors.append(f"too_few_citations: got {len(output.citations)}, min {min_citations}")
    return {
        "valid": len(missing_ids) == 0 and not too_few,
        "missing_ids": missing_ids,
        "too_few_citations": too_few,
        "errors": errors,
    }
