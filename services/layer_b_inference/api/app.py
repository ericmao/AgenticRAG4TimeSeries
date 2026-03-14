"""
Optional FastAPI: GET /healthz, POST /inference/run (InferenceRequest body).
Requires: pip install fastapi uvicorn
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from fastapi import FastAPI, HTTPException
except ImportError:
    FastAPI = None
    HTTPException = None

from contracts import InferenceRequest
from services.layer_b_inference.pipeline import events_from_episode_events, run_inference

app = FastAPI(title="Layer B Inference API") if FastAPI else None


if app is not None:
    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "service": "layer_b_inference"}

    @app.post("/inference/run")
    def inference_run(body: dict):
        """
        Accept InferenceRequest (JSON) with optional "events" key.
        If events provided, run inference and return InferenceResult (do not POST to Control Plane here).
        """
        try:
            request = InferenceRequest.model_validate(body.get("request", body))
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        events_raw = body.get("events", [])
        events = events_from_episode_events(events_raw)
        result = run_inference(request, events, fetch_time_ms=0.0)
        return result.model_dump()
