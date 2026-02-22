"""
Layer C: Policy-related contracts (Pydantic v2). Placeholder for policy rules and allowlists.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class PolicyAction(BaseModel):
    """Single policy action (allowlist)."""

    allowlist: Literal["block", "isolate", "watchlist", "collect_more_data"]
    target: Optional[str] = None
    params: dict[str, Any] = Field(default_factory=dict)


class PolicyRule(BaseModel):
    """Policy rule for scoping or constraints."""

    rule_id: str
    name: str
    conditions: list[str] = Field(default_factory=list)
    actions: list[PolicyAction] = Field(default_factory=list)
