"""
Pluggable execution fabric stub (OpenClaw-style).
Replace TaskDispatcher with a real remote executor when integrated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class AgentTask:
    """Unit of work for an agent step."""

    task_id: str
    agent_id: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTaskResult:
    ok: bool
    result: Any = None
    error: Optional[str] = None


class OpenClawAdapter:
    """
    Stub: synchronously runs a local runner(agent_id, payload) if registered.
    """

    def __init__(self) -> None:
        self._runners: dict[str, Callable[[dict[str, Any]], Any]] = {}

    def register(self, agent_id: str, fn: Callable[[dict[str, Any]], Any]) -> None:
        self._runners[agent_id] = fn

    def submit(self, task: AgentTask) -> AgentTaskResult:
        fn = self._runners.get(task.agent_id)
        if fn is None:
            return AgentTaskResult(ok=False, error=f"no runner for {task.agent_id}")
        try:
            out = fn(task.payload)
            return AgentTaskResult(ok=True, result=out)
        except Exception as e:
            return AgentTaskResult(ok=False, error=str(e))
