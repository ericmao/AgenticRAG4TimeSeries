"""
Wazuh Indexer（OpenSearch）查詢與 hit → event / episode 建構。
供 mvp_wazuh_episode_pg、wazuh_ingest_poll 共用。
"""
from __future__ import annotations

import base64
import json
import re
import ssl
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.request import Request, urlopen

from src.contracts.episode import Episode


def ts_string_to_ms(ts: Any) -> int:
    if ts is None:
        return 0
    if isinstance(ts, (int, float)):
        v = int(ts)
        if v < 1e12:
            v *= 1000
        return v
    s = str(ts).strip()
    if not s:
        return 0
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    if s.endswith("+0000"):
        s = s[:-5] + "+00:00"
    elif s.endswith("-0000"):
        s = s[:-5] + "-00:00"
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        return 0


def ms_to_iso_utc(ms: int) -> str:
    dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def safe_episode_id(ip: str, t0_ms: int) -> str:
    safe = re.sub(r"[^\w\-]", "-", ip.replace(".", "-"))
    return f"wazuh-{safe}-{t0_ms}"


def indexer_post(
    base_url: str,
    path: str,
    body: dict[str, Any],
    user: str,
    password: str,
    verify_ssl: bool,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + path
    data = json.dumps(body).encode("utf-8")
    req = Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    token = base64.b64encode(f"{user}:{password}".encode()).decode("ascii")
    req.add_header("Authorization", f"Basic {token}")
    ctx: Optional[ssl.SSLContext] = None
    if url.startswith("https"):
        if verify_ssl:
            ctx = ssl.create_default_context()
        else:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
    with urlopen(req, context=ctx, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def build_ip_query(
    target_ip: str,
    hours: int,
    size: int,
    match_all: bool = False,
) -> dict[str, Any]:
    range_clause = {
        "range": {
            "timestamp": {
                "gte": f"now-{hours}h",
                "lte": "now",
            }
        }
    }
    must: list[dict[str, Any]] = [range_clause]
    if not match_all:
        must.append(
            {
                "bool": {
                    "should": [
                        {"term": {"data.srcip": target_ip}},
                        {"term": {"data.dstip": target_ip}},
                        {"term": {"agent.ip": target_ip}},
                        {"wildcard": {"data.srcip": f"*{target_ip}*"}},
                        {"wildcard": {"data.dstip": f"*{target_ip}*"}},
                        {
                            "query_string": {
                                "query": target_ip.replace(":", "\\:"),
                                "default_field": "*",
                            }
                        },
                    ],
                    "minimum_should_match": 1,
                }
            }
        )
    return {
        "size": min(max(size, 1), 10000),
        "sort": [{"timestamp": {"order": "asc"}}],
        "query": {"bool": {"must": must}},
    }


def build_ip_query_after_ms(
    target_ip: str,
    after_ms: int,
    size: int,
    match_all: bool = False,
) -> dict[str, Any]:
    """自 after_ms 之後至現在（用於輪詢游標）。"""
    range_clause = {
        "range": {
            "timestamp": {
                "gt": ms_to_iso_utc(after_ms),
                "lte": "now",
            }
        }
    }
    must: list[dict[str, Any]] = [range_clause]
    if not match_all:
        must.append(
            {
                "bool": {
                    "should": [
                        {"term": {"data.srcip": target_ip}},
                        {"term": {"data.dstip": target_ip}},
                        {"term": {"agent.ip": target_ip}},
                        {"wildcard": {"data.srcip": f"*{target_ip}*"}},
                        {"wildcard": {"data.dstip": f"*{target_ip}*"}},
                        {
                            "query_string": {
                                "query": target_ip.replace(":", "\\:"),
                                "default_field": "*",
                            }
                        },
                    ],
                    "minimum_should_match": 1,
                }
            }
        )
    return {
        "size": min(max(size, 1), 10000),
        "sort": [{"timestamp": {"order": "asc"}}],
        "query": {"bool": {"must": must}},
    }


def hit_to_event(hit: dict[str, Any]) -> dict[str, Any]:
    src = hit.get("_source") or {}
    agent = src.get("agent") or {}
    rule = src.get("rule") or {}
    data = src.get("data") if isinstance(src.get("data"), dict) else {}
    ts_raw = src.get("timestamp") or src.get("@timestamp")
    ts_ms = ts_string_to_ms(ts_raw)
    agent_name = str(agent.get("name") or agent.get("id") or "unknown")
    rule_id = str(rule.get("id") or "")
    desc = str(rule.get("description") or rule_id or "alert")
    level = int(rule.get("level") or 0)
    conf = min(1.0, max(0.0, level / 15.0)) if level else 0.5
    groups = rule.get("groups") or []
    group_strs = [str(g) for g in groups] if isinstance(groups, list) else []
    mitre_raw = rule.get("mitre")
    mitre_hit = bool(mitre_raw) if isinstance(mitre_raw, list) else bool(mitre_raw)
    domain = str(groups[0]) if isinstance(groups, list) and groups else "internal"
    artifact: dict[str, Any] = {
        "host": agent_name,
        "rule_id": rule_id,
    }
    if agent.get("ip"):
        artifact["agent_ip"] = str(agent["ip"])
    for k in ("srcip", "dstip", "srcport", "dstport", "protocol"):
        if data.get(k) is not None:
            artifact[k] = data[k]
    return {
        "ts_ms": ts_ms,
        "entity": agent_name,
        "action": desc[:500],
        "artifact": artifact,
        "source": "wazuh",
        "confidence": conf,
        "domain": domain[:120],
        "event_id": str(hit.get("_id") or ""),
        "rule_level": level,
        "rule_groups": group_strs,
        "mitre_hit": mitre_hit,
    }


def build_episode_from_events(
    target_ip: str,
    run_id: str,
    events: list[dict[str, Any]],
    match_all: bool = False,
) -> tuple[Episode, str]:
    if not events:
        raise ValueError("no events from indexer")
    times = [int(e["ts_ms"]) for e in events if e.get("ts_ms")]
    if not times:
        t0_ms = 0
        t1_ms = 0
    else:
        t0_ms = min(times)
        t1_ms = max(times)
    if t1_ms < t0_ms:
        t0_ms, t1_ms = t1_ms, t0_ms
    episode_id = safe_episode_id(target_ip, t0_ms)
    entities = sorted({e.get("entity", "") for e in events if e.get("entity")})
    arts: list[dict[str, Any]] = [{"type": "ip", "value": target_ip}]
    ep = Episode(
        episode_id=episode_id,
        run_id=run_id,
        t0_ms=t0_ms,
        t1_ms=t1_ms,
        tenant_id="tenant-default",
        entity_id=entities[0] if entities else target_ip,
        window={"start": t0_ms, "end": t1_ms},
        entities=entities,
        artifacts=arts,
        sequence_tags=(["wazuh", "ip-scope", "match-all-smoke"] if match_all else ["wazuh", "ip-scope"]),
        events=events,
        risk_context=None,
    )
    return ep, episode_id


def max_event_ts_ms(events: list[dict[str, Any]]) -> int:
    ts = [int(e["ts_ms"]) for e in events if e.get("ts_ms")]
    return max(ts) if ts else 0
