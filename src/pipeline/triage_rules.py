"""
解析多規則 triage 的 rule_id 列表（與 episode.sequence_tags 交集或明確指定）。
"""
from __future__ import annotations

import os
import re
from typing import Optional

from src.contracts.episode import Episode

# 與 sequence_tags 交集時使用的候選（小寫比對）
_DEFAULT_TAG_CANDIDATES = frozenset(
    {
        "lateral",
        "burst",
        "exfil",
        "logon",
        "logoff",
        "login",
        "device",
        "suspicious",
        "anomaly",
        "critical",
    }
)


def sanitize_rule_id_for_key(rule_id: str) -> str:
    """檔名／鍵名安全：英數與底線。"""
    s = rule_id.strip().lower().replace("-", "_")
    s = re.sub(r"[^a-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "default"


def resolve_triage_rules(
    episode: Episode,
    explicit: Optional[list[str]] = None,
) -> list[str]:
    """
    若 explicit 非空（或環境變數 TRIAGE_RULES 逗號分隔）則使用；
    否則取 episode.sequence_tags 與候選集合交集（保留原順序）；
    若仍為空則 [\"default\"]。
    """
    if explicit:
        rules = [sanitize_rule_id_for_key(x) for x in explicit if x and str(x).strip()]
        return list(dict.fromkeys(rules)) or ["default"]

    env = os.environ.get("TRIAGE_RULES", "").strip()
    if env:
        parts = [sanitize_rule_id_for_key(p) for p in env.split(",") if p.strip()]
        return list(dict.fromkeys(parts)) or ["default"]

    tags = [str(t).lower() for t in (episode.sequence_tags or [])]
    matched: list[str] = []
    seen: set[str] = set()
    for t in tags:
        if t in _DEFAULT_TAG_CANDIDATES and t not in seen:
            matched.append(sanitize_rule_id_for_key(t))
            seen.add(t)
    if matched:
        return matched
    return ["default"]
