"""agent_id -> callable for use with TaskDispatcher / OpenClawAdapter."""
from __future__ import annotations

from typing import Any, Callable


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, Callable[..., Any]] = {}

    def register(self, agent_id: str, fn: Callable[..., Any]) -> None:
        self._agents[agent_id] = fn

    def get(self, agent_id: str) -> Callable[..., Any] | None:
        return self._agents.get(agent_id)

    def list_ids(self) -> list[str]:
        return sorted(self._agents.keys())
