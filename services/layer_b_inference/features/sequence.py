"""
Build sequence representation from NormalizedEvent list. MVP: ordered events + simple features.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from contracts import NormalizedEvent


def build_ordered_events(events: list[NormalizedEvent]) -> list[NormalizedEvent]:
    """Return events sorted by ts_ms."""
    return sorted(events, key=lambda e: e.ts_ms)


def build_features(events: list[NormalizedEvent]) -> dict[str, Any]:
    """
    MVP features: event_type transitions, burstiness, unique destination counts.
    Returns dict for heuristic scorer.
    """
    if not events:
        return {
            "event_count": 0,
            "unique_entities": 0,
            "unique_destinations": 0,
            "action_counts": {},
            "transition_pairs": [],
            "burstiness": 0.0,
        }
    ordered = build_ordered_events(events)
    actions = [e.action for e in ordered]
    action_counts = dict(Counter(actions))
    entities = set(e.entity for e in ordered)
    destinations = set()
    for e in ordered:
        if isinstance(e.artifact, dict) and "host" in e.artifact:
            destinations.add(e.artifact["host"])
        elif isinstance(e.artifact, str):
            destinations.add(e.artifact)
    transition_pairs = []
    for i in range(len(actions) - 1):
        transition_pairs.append((actions[i], actions[i + 1]))
    # Simple burstiness: events per 5-min bucket max
    if not ordered:
        burstiness = 0.0
    else:
        t0 = ordered[0].ts_ms
        bucket_ms = 5 * 60 * 1000
        buckets: dict[int, int] = {}
        for e in ordered:
            b = (e.ts_ms - t0) // bucket_ms
            buckets[b] = buckets.get(b, 0) + 1
        burstiness = float(max(buckets.values())) if buckets else 0.0
    return {
        "event_count": len(events),
        "unique_entities": len(entities),
        "unique_destinations": len(destinations),
        "action_counts": action_counts,
        "transition_pairs": transition_pairs[:50],
        "burstiness": burstiness,
    }
