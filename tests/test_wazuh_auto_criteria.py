"""wazuh_auto_criteria 單元測試。"""
from __future__ import annotations

from src.integrations.wazuh_auto_criteria import (
    WazuhAutoCriteria,
    dedup_should_skip_last_run,
    should_create_episode,
)


def test_criteria_off_always_passes():
    crit = WazuhAutoCriteria(
        enabled=False,
        min_rule_level=99,
        min_alert_count=99,
        min_distinct_rule_ids=99,
        mitre_triggers=False,
    )
    events = [{"rule_level": 1, "artifact": {"rule_id": "a"}, "mitre_hit": False}]
    ok, d = should_create_episode(events, crit)
    assert ok is True
    assert d.get("mode") == "criteria_off"


def test_or_max_level():
    crit = WazuhAutoCriteria(
        enabled=True,
        min_rule_level=10,
        min_alert_count=99,
        min_distinct_rule_ids=99,
        mitre_triggers=False,
    )
    events = [{"rule_level": 11, "artifact": {"rule_id": "x"}, "mitre_hit": False}]
    ok, d = should_create_episode(events, crit)
    assert ok is True
    assert "max_rule_level" in d.get("matched_branches", [])


def test_reject_when_no_branch():
    crit = WazuhAutoCriteria(
        enabled=True,
        min_rule_level=15,
        min_alert_count=10,
        min_distinct_rule_ids=5,
        mitre_triggers=False,
    )
    events = [
        {"rule_level": 5, "artifact": {"rule_id": "a"}, "mitre_hit": False},
        {"rule_level": 6, "artifact": {"rule_id": "a"}, "mitre_hit": False},
    ]
    ok, d = should_create_episode(events, crit)
    assert ok is False
    assert d.get("reason") == "no_branch_matched"


def test_dedup_skip():
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    recent = now - timedelta(minutes=30)
    assert dedup_should_skip_last_run(recent, 60) is True
    assert dedup_should_skip_last_run(recent, 0) is False
    assert dedup_should_skip_last_run(None, 60) is False
