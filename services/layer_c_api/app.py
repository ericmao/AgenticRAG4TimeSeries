"""
Layer C HTTP API: run retrieve → analyze → writeback, driven by Control Plane (guacamole-ai).
GET /health, POST /analyze (body: episode_path or episode JSON).
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import sys
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Layer C Analysis API", version="0.1.0")


class AnalyzeRequest(BaseModel):
    """Request body: either episode_path (relative to repo root) or inline episode JSON."""
    episode_path: str | None = Field(None, description="Path to episode JSON, e.g. tests/demo/episode_insider_highrisk.json")
    episode: dict | None = Field(None, description="Inline episode JSON (used when episode_path is not set)")
    mode: str = Field("dry_run", description="Writeback mode: dry_run, review, auto")


def _run_layer_c_pipeline(episode_path: str | Path, mode: str = "dry_run") -> dict:
    """Run retrieve → analyze → writeback. Returns summary dict."""
    from src.contracts.episode import Episode
    from src.contracts.evidence import EvidenceSet
    from src.pipeline.retrieve_evidence import build_evidence_set
    from src.pipeline.validate_and_repair import run_analyze_with_validation
    from src.pipeline.writeback_pipeline import run_writeback

    root = Path(_REPO_ROOT)
    episode_path = Path(episode_path)
    if not episode_path.is_absolute():
        episode_path = root / episode_path
    if not episode_path.exists():
        raise FileNotFoundError(f"Episode file not found: {episode_path}")

    episode = Episode.model_validate(json.loads(episode_path.read_text(encoding="utf-8")))
    episode_id = episode.episode_id

    # 1) Retrieve
    build_evidence_set(str(episode_path), hypothesis_path=None)
    evidence_path = root / "outputs" / "evidence" / f"{episode_id}.json"
    evidence_set = EvidenceSet.model_validate(json.loads(evidence_path.read_text(encoding="utf-8")))

    # 2) Analyze
    outputs, status = run_analyze_with_validation(episode, evidence_set, trust_signals=None)
    if status != "ok":
        return {
            "ok": False,
            "episode_id": episode_id,
            "step": "analyze",
            "status": "validation_failed",
            "message": "Agent validation failed; see outputs/issues/",
        }

    # 3) Writeback (posts to Control Plane if CONTROL_PLANE_BASE_URL is set)
    patch, _ = run_writeback(str(episode_path), mode=mode)
    writeback_posted = bool(os.environ.get("CONTROL_PLANE_BASE_URL", "").strip())

    return {
        "ok": True,
        "episode_id": episode_id,
        "run_id": patch.run_id,
        "mode": patch.mode,
        "writeback_posted": writeback_posted,
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "layer_c_analysis"}


@app.post("/analyze")
def analyze(body: AnalyzeRequest):
    """
    Run Layer C pipeline: retrieve → analyze → writeback.
    Provide episode_path (path under repo root) or episode (inline JSON).
    Writeback is posted to Control Plane when CONTROL_PLANE_BASE_URL is set.
    """
    episode_path: str | None = None
    if body.episode_path:
        episode_path = body.episode_path
    elif body.episode:
        # Write inline episode to temp file under repo root so pipeline can resolve outputs/
        root = Path(_REPO_ROOT)
        root.mkdir(parents=True, exist_ok=True)
        fd, path = tempfile.mkstemp(suffix=".json", prefix="episode_", dir=root)
        try:
            os.write(fd, json.dumps(body.episode, ensure_ascii=False).encode("utf-8"))
            os.close(fd)
            episode_path = path
        except Exception as e:
            os.close(fd)
            raise HTTPException(status_code=400, detail=f"Invalid episode JSON: {e}")
    else:
        raise HTTPException(status_code=400, detail="Provide episode_path or episode")

    try:
        result = _run_layer_c_pipeline(episode_path, mode=body.mode)
        if result.get("ok"):
            return result
        raise HTTPException(status_code=422, detail=result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
