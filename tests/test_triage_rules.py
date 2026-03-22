"""resolve_triage_rules、santize_rule_id、writeback 展平鍵名慣例。"""
from __future__ import annotations

from src.contracts.episode import Episode
from src.pipeline.triage_rules import resolve_triage_rules, sanitize_rule_id_for_key


def _minimal_episode(**kwargs) -> Episode:
    base = {
        "episode_id": "ep1",
        "run_id": "r1",
        "t0_ms": 0,
        "t1_ms": 1,
    }
    base.update(kwargs)
    return Episode.model_validate(base)


def test_sanitize_rule_id():
    assert sanitize_rule_id_for_key("Lateral-Move") == "lateral_move"
    assert sanitize_rule_id_for_key("  ") == "default"


def test_resolve_explicit():
    ep = _minimal_episode(sequence_tags=["lateral"])
    assert resolve_triage_rules(ep, explicit=["alpha", "beta"]) == ["alpha", "beta"]


def test_resolve_from_tags():
    ep = _minimal_episode(sequence_tags=["lateral", "logon", "unknown_tag"])
    r = resolve_triage_rules(ep, explicit=None)
    assert "lateral" in r and "logon" in r
    assert "unknown_tag" not in r


def test_resolve_default_when_empty_tags():
    ep = _minimal_episode(sequence_tags=[])
    assert resolve_triage_rules(ep, explicit=None) == ["default"]
