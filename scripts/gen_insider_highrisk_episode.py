#!/usr/bin/env python3
"""
Generate a high-risk insider Episode JSON for Layer C demo.
Deterministic: same args produce same output.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Repo root on path for src.contracts
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.contracts.episode import Episode


def _event(
    ts_ms: int,
    entity: str,
    action: str,
    host: str,
    source: str,
    confidence: float = 0.9,
    domain: str = "IT",
) -> dict:
    return {
        "ts_ms": ts_ms,
        "entity": entity,
        "action": action,
        "artifact": {"host": host},
        "source": source,
        "confidence": confidence,
        "domain": domain,
    }


def build_events(
    t0_ms: int,
    window_ms: int,
    user: str,
    hosts: list[str],
    burst_events: int,
    device_churn: int,
) -> list[dict]:
    t1_ms = t0_ms + window_ms
    events: list[dict] = []
    step_min = 60_000  # 1 min between initial logons

    # a) 4 initial logon events across 4 hosts (lateral trigger)
    for i, host in enumerate(hosts[:4]):
        ts = t0_ms + i * step_min
        events.append(_event(ts, user, "logon", host, "logon"))

    # b) burst_events alternating logon/logoff over the window (same user, rotating hosts)
    # Place in second half of window to avoid overlap with device churn
    half = window_ms // 2
    start_burst = t0_ms + half
    burst_step = max(1, (half - 1) // burst_events) if burst_events else 0
    for i in range(burst_events):
        ts = start_burst + i * burst_step
        if ts >= t1_ms:
            break
        action = "logon" if i % 2 == 0 else "logoff"
        host = hosts[i % len(hosts)]
        events.append(_event(ts, user, action, host, "logon"))

    # c) device_churn connect/disconnect pairs (same user, host rotates)
    start_dev = t0_ms + 4 * step_min + 10_000
    dev_step = 5_000
    for j in range(device_churn * 2):
        ts = start_dev + j * dev_step
        if ts >= t1_ms:
            break
        action = "connect" if j % 2 == 0 else "disconnect"
        host = hosts[(j // 2) % len(hosts)]
        events.append(_event(ts, user, action, host, "device"))

    events.sort(key=lambda e: (e["ts_ms"], e.get("action", "")))
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate high-risk insider Episode JSON")
    parser.add_argument("--out_path", default="tests/demo/episode_insider_highrisk.json", help="Output JSON path")
    parser.add_argument("--user", default="USER0420", help="User entity")
    parser.add_argument("--hosts", default="PC010,PC011,PC012,PC013", help="Comma-separated host list")
    parser.add_argument("--t0_ms", type=int, default=1771754000000, help="Window start (epoch ms)")
    parser.add_argument("--window_ms", type=int, default=3600000, help="Window length (ms)")
    parser.add_argument("--burst_events", type=int, default=60, help="Number of alternating logon/logoff events")
    parser.add_argument("--device_churn", type=int, default=12, help="Number of connect/disconnect pairs")
    args = parser.parse_args()

    user = args.user.strip()
    hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
    if len(hosts) < 4:
        hosts = (hosts * 4)[:4]
    t0_ms = args.t0_ms
    window_ms = args.window_ms
    t1_ms = t0_ms + window_ms

    events = build_events(t0_ms, window_ms, user, hosts, args.burst_events, args.device_churn)
    entities = [user] + hosts
    artifacts = [{"type": "user", "value": user}] + [{"type": "host", "value": h} for h in hosts]
    sequence_tags = ["logon", "device", "lateral", "burst"]
    episode_id = f"cert-{user}-highrisk"
    run_id = "cert-run-hr-1"

    episode = Episode(
        episode_id=episode_id,
        run_id=run_id,
        t0_ms=t0_ms,
        t1_ms=t1_ms,
        entities=entities,
        artifacts=artifacts,
        sequence_tags=sequence_tags,
        events=events,
    )
    out_path = Path(args.out_path)
    if not out_path.is_absolute():
        out_path = _REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(episode.model_dump_json(indent=2), encoding="utf-8")
    print(f"Wrote {out_path} (episode_id={episode_id}, events={len(events)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
