"""
Merge KB + OpenCTI candidates, deduplicate by stable hash, rank, cap to max_items. Fill EvidenceSet.stats.
"""
from __future__ import annotations

import hashlib
from typing import Any, List, Optional

from src.contracts.evidence import EvidenceItem, EvidenceSet


def _stable_hash(item: EvidenceItem) -> str:
    """Stable hash over (source, kind, title, body, stix_id, chunk_id) for deduplication."""
    payload = "|".join([
        item.source,
        item.kind,
        item.title,
        item.body,
        item.stix_id or "",
        item.chunk_id or "",
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_weight(source: str) -> int:
    """Higher = better. opencti > kb > taxii > other."""
    return {"opencti": 3, "kb": 2, "taxii": 1, "other": 0}.get(source, 0)


def _sort_key(item: EvidenceItem) -> tuple:
    """Rank: score desc, source weight desc, ts_ms desc (None last)."""
    ts = item.ts_ms if item.ts_ms is not None else -1
    return (-item.score, -_source_weight(item.source), -ts)


def assemble_evidence(
    kb_items: List[EvidenceItem],
    opencti_items: List[EvidenceItem],
    episode_id: str,
    run_id: str,
    max_items: int = 50,
) -> EvidenceSet:
    """
    Merge, deduplicate by stable hash, rank by score then source weight then recency, cap to max_items.
    Fill stats: counts per source/kind, deduped_count, capped_count.
    """
    seen_hashes: set[str] = set()
    merged: list[EvidenceItem] = []
    for item in kb_items + opencti_items:
        h = _stable_hash(item)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        merged.append(item)

    merged.sort(key=_sort_key)
    capped = merged[:max_items]

    # Stats
    source_counts: dict[str, int] = {}
    kind_counts: dict[str, int] = {}
    for item in capped:
        source_counts[item.source] = source_counts.get(item.source, 0) + 1
        kind_counts[item.kind] = kind_counts.get(item.kind, 0) + 1
    total_candidates = len(kb_items) + len(opencti_items)
    stats: dict[str, Any] = {
        "total_candidates": total_candidates,
        "deduped_count": len(merged),
        "capped_count": len(capped),
        "by_source": source_counts,
        "by_kind": kind_counts,
    }

    return EvidenceSet(
        episode_id=episode_id,
        run_id=run_id,
        items=capped,
        stats=stats,
        max_items=max_items,
    )
