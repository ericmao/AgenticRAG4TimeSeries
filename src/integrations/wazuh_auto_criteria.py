"""
Wazuh 輪詢自動產生 episode 的行為準則（可環境變數設定）。

預設關閉（與舊行為相容：每批 alerts 都產生 episode）。啟用後，
須符合下列**任一**條件才產生 episode（OR）：

1. 單筆 rule_level ≥ WAZUH_AUTO_MIN_RULE_LEVEL（0 表示關閉此條）
2. 本批 alert 筆數 ≥ WAZUH_AUTO_MIN_ALERT_COUNT（0 表示關閉）
3. 本批不同 rule.id 數 ≥ WAZUH_AUTO_MIN_DISTINCT_RULES（0 表示關閉）
4. WAZUH_AUTO_MITRE_TRIGGERS=1 且任一有 MITRE 標記（mitre_hit）

macOS / Windows：僅使用 Python 標準庫與 os.environ，路徑由呼叫端以 pathlib 處理。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _int_env(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        return default
    try:
        return int(str(v).strip())
    except ValueError:
        return default


@dataclass(frozen=True)
class WazuhAutoCriteria:
    """自動 episode 門檻；任一條件滿足即通過（OR）。"""

    enabled: bool
    min_rule_level: int
    min_alert_count: int
    min_distinct_rule_ids: int
    mitre_triggers: bool

    @classmethod
    def from_env(cls) -> WazuhAutoCriteria:
        return cls(
            enabled=_bool_env("WAZUH_AUTO_EPISODE_CRITERIA", default=False),
            min_rule_level=_int_env("WAZUH_AUTO_MIN_RULE_LEVEL", 12),
            min_alert_count=_int_env("WAZUH_AUTO_MIN_ALERT_COUNT", 5),
            min_distinct_rule_ids=_int_env("WAZUH_AUTO_MIN_DISTINCT_RULES", 3),
            mitre_triggers=_bool_env("WAZUH_AUTO_MITRE_TRIGGERS", default=True),
        )


def _distinct_rule_ids(events: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for e in events:
        art = e.get("artifact") if isinstance(e.get("artifact"), dict) else {}
        rid = str(art.get("rule_id") or "").strip()
        if rid:
            out.add(rid)
    return out


def dedup_minutes_from_env() -> int:
    """WAZUH_AUTO_DEDUP_MINUTES：同一 target_ip 連續兩次分析的最短間隔（分鐘），0=關閉。"""
    return _int_env("WAZUH_AUTO_DEDUP_MINUTES", 0)


def should_create_episode(
    events: list[dict[str, Any]],
    criteria: WazuhAutoCriteria,
) -> tuple[bool, dict[str, Any]]:
    """
    若 criteria.enabled 為 False，一律通過。
    否則依 OR 規則判斷；回傳 (通過與否, 診斷 dict)。
    """
    if not criteria.enabled:
        return True, {"mode": "criteria_off", "pass": True}

    if not events:
        return False, {"mode": "criteria_on", "pass": False, "reason": "no_events"}

    levels = [int(e.get("rule_level") or 0) for e in events]
    max_level = max(levels) if levels else 0
    n = len(events)
    distinct = _distinct_rule_ids(events)
    any_mitre = any(bool(e.get("mitre_hit")) for e in events)

    branches: list[str] = []

    if criteria.min_rule_level > 0 and max_level >= criteria.min_rule_level:
        branches.append("max_rule_level")
    if criteria.min_alert_count > 0 and n >= criteria.min_alert_count:
        branches.append("alert_count")
    if criteria.min_distinct_rule_ids > 0 and len(distinct) >= criteria.min_distinct_rule_ids:
        branches.append("distinct_rules")
    if criteria.mitre_triggers and any_mitre:
        branches.append("mitre")

    ok = len(branches) > 0
    detail: dict[str, Any] = {
        "mode": "criteria_on",
        "pass": ok,
        "max_rule_level": max_level,
        "alert_count": n,
        "distinct_rule_ids": len(distinct),
        "any_mitre": any_mitre,
        "matched_branches": branches,
    }
    if not ok:
        detail["reason"] = "no_branch_matched"
    return ok, detail


def dedup_should_skip_last_run(last_created_at: Optional[Any], dedup_minutes: int) -> bool:
    """
    若上次同一 target_ip 的 wazuh run 在 dedup_minutes 內，則跳過產生 episode（仍應前進 Indexer 游標）。
    dedup_minutes <= 0 表示關閉。
    """
    if dedup_minutes <= 0 or last_created_at is None:
        return False
    now = datetime.now(timezone.utc)
    last = last_created_at
    if isinstance(last, datetime):
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        return (now - last).total_seconds() < dedup_minutes * 60
    return False
