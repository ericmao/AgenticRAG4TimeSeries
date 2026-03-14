"""
POST InferenceResult to Control Plane. Retry with exponential backoff; store failed in outbox.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from contracts import InferenceResult


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _base_url() -> str:
    return os.environ.get("CONTROL_PLANE_BASE_URL", "").strip()


def _token() -> str:
    return os.environ.get("CONTROL_PLANE_TOKEN", "").strip()


def post_result(result: InferenceResult) -> tuple[bool, Optional[str]]:
    """
    POST result to {CONTROL_PLANE_BASE_URL}/api/v1/inference/results.
    Returns (success, error_message).
    """
    base = _base_url()
    token = _token()
    if not base or not token:
        return False, "CONTROL_PLANE_BASE_URL or CONTROL_PLANE_TOKEN not set"
    url = f"{base.rstrip('/')}/api/v1/inference/results"
    payload = result.model_dump_json()
    # Retry with exponential backoff
    for attempt in range(3):
        try:
            import urllib.request
            req = urllib.request.Request(
                url,
                data=payload.encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                if 200 <= resp.status < 300:
                    return True, None
                return False, f"HTTP {resp.status}"
        except Exception as e:
            if attempt == 2:
                return False, str(e)
            time.sleep(2 ** attempt)
    return False, "max retries exceeded"


def save_outbox(result: InferenceResult, error: str) -> Path:
    """Append failed result to outbox file for later retry."""
    root = _repo_root()
    outbox_dir = root / "outputs" / "layer_b_outbox"
    outbox_dir.mkdir(parents=True, exist_ok=True)
    path = outbox_dir / "failed_results.jsonl"
    line = json.dumps({"result": result.model_dump(), "error": error}) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
    return path
