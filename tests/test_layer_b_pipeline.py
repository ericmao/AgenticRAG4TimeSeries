"""One inference pipeline test: run_inference produces valid InferenceResult."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contracts import InferenceRequest, NormalizedEvent
from services.layer_b_inference.pipeline import run_inference


def test_run_inference_returns_result():
    request = InferenceRequest(
        job_id="test-job",
        tenant_id="tenant-1",
        endpoint_id="ep-1",
        t0_ms=1700000000000,
        t1_ms=1700003600000,
    )
    events = [
        NormalizedEvent(ts_ms=1700001000000, entity="user-001", action="logon", artifact={"host": "PC01"}),
        NormalizedEvent(ts_ms=1700002000000, entity="user-001", action="logoff", artifact={"host": "PC01"}),
        NormalizedEvent(ts_ms=1700002500000, entity="user-001", action="connect", artifact={"host": "PC02"}),
    ]
    result = run_inference(request, events, fetch_time_ms=10.0)
    assert result.tenant_id == "tenant-1"
    assert result.hypothesis.hypothesis_id
    assert result.status == "success"
    assert "fetch_time_ms" in result.metrics
    assert "inference_time_ms" in result.metrics
