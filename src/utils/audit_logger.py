"""
Audit logging: JSONL to outputs/audit/<run_id>.jsonl.
"""
import json
import os
from pathlib import Path
from typing import Any, Optional

# Repo root: parent of src/
_REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = _REPO_ROOT / "outputs" / "audit"


def audit_log(
    event: dict,
    run_id: str,
    episode_id: Optional[str],
    component: str,
    prompt_version: str = "v0.1",
) -> None:
    """Append one JSONL record to outputs/audit/<run_id>.jsonl."""
    import time
    record = {
        "ts_ms": int(time.time() * 1000),
        "run_id": run_id,
        "episode_id": episode_id,
        "component": component,
        "payload": event,
        "prompt_version": prompt_version,
    }
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    path = AUDIT_DIR / f"{run_id}.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
