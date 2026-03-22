"""MVP UI 背景管線（Wazuh / CERT）完成狀態，供前端輪詢後顯示結果。"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


_JOB_ID_RE = re.compile(r"^[a-f0-9]{32}$")


def validate_job_id(job_id: str) -> bool:
    return bool(job_id and _JOB_ID_RE.match(job_id))


def job_dir(repo_root: Path) -> Path:
    d = repo_root / "outputs" / "mvp_ui_jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def job_file(repo_root: Path, job_id: str) -> Path:
    if not validate_job_id(job_id):
        raise ValueError("invalid job_id")
    return job_dir(repo_root) / f"{job_id}.json"


def write_job_running(
    repo_root: Path,
    job_id: str,
    kind: str,
    request: dict[str, Any],
) -> None:
    p = job_file(repo_root, job_id)
    payload: dict[str, Any] = {
        "job_id": job_id,
        "kind": kind,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "request": request,
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def finalize_job(
    repo_root: Path,
    job_id: str,
    *,
    kind: str,
    request: dict[str, Any],
    exit_code: int,
    stdout: str = "",
    stderr: str = "",
    error: Optional[str] = None,
) -> None:
    p = job_file(repo_root, job_id)
    status = "ok" if exit_code == 0 and not error else "failed"
    payload: dict[str, Any] = {
        "job_id": job_id,
        "kind": kind,
        "status": status,
        "exit_code": exit_code,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "request": request,
        "stdout": (stdout or "")[:16000],
        "stderr": (stderr or "")[:16000],
    }
    if error:
        payload["error"] = error[:4000]
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_job(repo_root: Path, job_id: str) -> Optional[dict[str, Any]]:
    if not validate_job_id(job_id):
        return None
    p = job_file(repo_root, job_id)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
