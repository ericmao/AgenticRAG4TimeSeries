"""Agent catalog + read outputs/agent_activity.json for dashboard."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# SenseL EvidenceOps + legacy Layer C agent ids surfaced in UI
AGENT_CATALOG: list[dict[str, str]] = [
    {
        "agent_id": "kb_describer",
        "title": "KB Describer",
        "description": "監看 KB 新檔／變更，以 Ollama 寫入單檔 LLM 概要（scripts/kb_watch_llm.py 輪詢；OpenClaw agent_id kb_describer）",
    },
    {"agent_id": "triage", "title": "Triage", "description": "嚴重度與優先序（僅依 EvidenceSet）"},
    {"agent_id": "entity_investigation", "title": "Entity Investigation", "description": "實體與證據對齊"},
    {"agent_id": "cti_correlation", "title": "CTI Correlation", "description": "證據內 IOC/CTI 型態叢集（無外部查詢）"},
    {"agent_id": "hunt_planner", "title": "Hunt Planner", "description": "獵捕查詢／軸轉建議"},
    {"agent_id": "response_advisor", "title": "Response Advisor", "description": "回應動作與範圍（policy 檢查）"},
]


def read_agent_activity(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "outputs" / "agent_activity.json"
    if not path.is_file():
        return {
            "schema": "agent_activity.v1",
            "overall_status": "unknown",
            "current_agent_id": None,
            "current_detail": "尚無快照：請在本機執行 EvidenceOps orchestrator（會寫入 outputs/agent_activity.json）",
            "episode_id": None,
            "run_id": None,
            "steps": [],
            "updated_at_ms": None,
        }
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {
            "schema": "agent_activity.v1",
            "overall_status": "error",
            "current_detail": str(e),
            "steps": [],
        }
