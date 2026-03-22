"""Routing policy for EvidenceOps."""
from __future__ import annotations

from src.layer_c.orchestrator.routing_policy import plan_route, triage_level_from_output


def test_triage_level_from_output():
    assert triage_level_from_output({"triage_level": "critical"}) == "critical"
    assert triage_level_from_output({}) == "suspicious"


def test_plan_route_noise_skips_cti_and_hunt():
    r = plan_route("noise")
    assert r["run_cti_correlation"] is False
    assert r["run_hunt_planner"] is False
    assert r["run_response_advisor"] is True


def test_plan_route_critical_full():
    r = plan_route("critical")
    assert r["run_cti_correlation"] is True
    assert r["run_hunt_planner"] is True
