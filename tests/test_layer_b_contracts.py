"""Schema validation for Layer B contracts."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contracts import Hypothesis, InferenceRequest, InferenceResult, NormalizedEvent


def test_inference_request():
    req = InferenceRequest(
        job_id="j1",
        tenant_id="t1",
        endpoint_id="e1",
        t0_ms=1000,
        t1_ms=2000,
    )
    assert req.tenant_id == "t1"
    assert req.t1_ms == 2000


def test_normalized_event():
    e = NormalizedEvent(ts_ms=1000, entity="u1", action="logon", artifact={"host": "PC01"})
    assert e.source == "logon"
    assert e.artifact["host"] == "PC01"


def test_hypothesis():
    h = Hypothesis(hypothesis_id="h1", text="Test", suspected_tactics=["T1021"])
    assert h.suspected_tactics == ["T1021"]


def test_inference_result():
    h = Hypothesis(hypothesis_id="h1", text="Test")
    r = InferenceResult(tenant_id="t1", endpoint_id="e1", hypothesis=h, status="success")
    assert r.status == "success"
    assert r.hypothesis.hypothesis_id == "h1"
