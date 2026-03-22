"""Dispatch AgentTask through OpenClawAdapter or direct registry."""
from __future__ import annotations

from typing import Any, Optional

from src.layer_c.runtime.agent_registry import AgentRegistry
from src.layer_c.runtime.openclaw_adapter import AgentTask, AgentTaskResult, OpenClawAdapter


class TaskDispatcher:
    def __init__(self, adapter: Optional[OpenClawAdapter] = None, registry: Optional[AgentRegistry] = None):
        self.adapter = adapter or OpenClawAdapter()
        self.registry = registry or AgentRegistry()

    def dispatch(self, task: AgentTask) -> AgentTaskResult:
        fn = self.registry.get(task.agent_id)
        if fn is not None:
            self.adapter.register(task.agent_id, lambda p: fn(**p))
        return self.adapter.submit(task)
