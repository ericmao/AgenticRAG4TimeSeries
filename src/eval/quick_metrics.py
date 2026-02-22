"""
Optional: write a small metrics JSON (e.g. after analyze) for eval.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def write_quick_metrics(
    episode_id: str,
    run_id: str,
    status: str,
    outputs: dict[str, Any],
) -> Path:
    """Write outputs/eval/<episode_id>_metrics.json with status and per-agent citation count / confidence."""
    root = _repo_root()
    out_dir = root / "outputs" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{episode_id}_metrics.json"
    metrics = {
        "episode_id": episode_id,
        "run_id": run_id,
        "status": status,
        "agents": {
            aid: {"citations": len(out.get("citations", [])), "confidence": out.get("confidence")}
            for aid, out in outputs.items()
        },
    }
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path
