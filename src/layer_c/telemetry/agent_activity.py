"""
Cross-process agent activity snapshot for MVP UI / dashboard.
Writes outputs/agent_activity.json (repo root) so Docker bind-mount sees updates.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional


def activity_path(repo_root: Path) -> Path:
    repo_root.mkdir(parents=True, exist_ok=True)
    out = repo_root / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out / "agent_activity.json"


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, ensure_ascii=False, indent=2)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(raw, encoding="utf-8")
    tmp.replace(path)


def reset_run(repo_root: Path, episode_id: str, run_id: str) -> None:
    """Call at orchestrator start."""
    now = int(time.time() * 1000)
    _atomic_write(
        activity_path(repo_root),
        {
            "schema": "agent_activity.v1",
            "updated_at_ms": now,
            "orchestrator": "evidenceops",
            "episode_id": episode_id,
            "run_id": run_id,
            "overall_status": "running",
            "current_agent_id": None,
            "current_detail": "orchestrator started",
            "steps": [],
        },
    )


def record_agent_start(repo_root: Path, agent_id: str, detail: str = "") -> None:
    path = activity_path(repo_root)
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    now = int(time.time() * 1000)
    data["updated_at_ms"] = now
    data["current_agent_id"] = agent_id
    data["current_detail"] = detail or f"running {agent_id}"
    data["overall_status"] = "running"
    steps = data.get("steps") or []
    steps.append(
        {
            "agent_id": agent_id,
            "phase": "running",
            "started_at_ms": now,
            "detail": data["current_detail"],
        }
    )
    data["steps"] = steps
    _atomic_write(path, data)


def record_agent_done(
    repo_root: Path,
    agent_id: str,
    detail: str = "",
    *,
    skipped: bool = False,
) -> None:
    path = activity_path(repo_root)
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    now = int(time.time() * 1000)
    data["updated_at_ms"] = now
    data["current_agent_id"] = None
    data["current_detail"] = detail or ("skipped" if skipped else f"finished {agent_id}")
    steps = data.get("steps") or []
    steps.append(
        {
            "agent_id": agent_id,
            "phase": "skipped" if skipped else "done",
            "finished_at_ms": now,
            "detail": data["current_detail"],
        }
    )
    data["steps"] = steps
    _atomic_write(path, data)


def finalize_run(repo_root: Path, ok: bool, message: str = "") -> None:
    path = activity_path(repo_root)
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    now = int(time.time() * 1000)
    data["updated_at_ms"] = now
    data["overall_status"] = "completed" if ok else "failed"
    data["current_agent_id"] = None
    data["current_detail"] = message or ("validation ok" if ok else "validation failed")
    data["finished_at_ms"] = now
    _atomic_write(path, data)


def idle_snapshot(repo_root: Path, message: str = "idle") -> None:
    """No run in progress (e.g. after finalize or startup)."""
    now = int(time.time() * 1000)
    _atomic_write(
        activity_path(repo_root),
        {
            "schema": "agent_activity.v1",
            "updated_at_ms": now,
            "orchestrator": "evidenceops",
            "episode_id": None,
            "run_id": None,
            "overall_status": "idle",
            "current_agent_id": None,
            "current_detail": message,
            "steps": [],
        },
    )
