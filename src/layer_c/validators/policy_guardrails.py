"""Wrapper for response advisor policy checks."""
from __future__ import annotations

from typing import Any

from src.contracts.agent_output import AgentOutput
from src.contracts.episode import Episode
from src.validators.policy_guardrails import validate_response_advisor_policy as _validate


def validate_response_policy(output: AgentOutput, episode: Episode) -> dict[str, Any]:
    return _validate(output, episode)
