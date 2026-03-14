"""
Build retrieval queries from Episode and optional Hypothesis. Deterministic ordering.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from src.contracts.episode import Episode, Hypothesis


# Artifact type keys for structured terms
ARTIFACT_IP = "ip"
ARTIFACT_DOMAIN = "domain"
ARTIFACT_HASH = "hash"
ARTIFACT_PROTOCOL_CMD = "protocol_cmd"
ARTIFACT_TOPIC = "topic"
ARTIFACT_FILE = "file"


def _artifact_type(t: str) -> str:
    """Normalize artifact type for query terms."""
    t = (t or "").strip().lower()
    if t in ("ip", "ipv4", "ipv6", "address"):
        return ARTIFACT_IP
    if t in ("domain", "hostname", "fqdn"):
        return ARTIFACT_DOMAIN
    if t in ("hash", "sha256", "sha1", "md5", "file_hash"):
        return ARTIFACT_HASH
    if t in ("protocol_cmd", "cmd", "command", "protocol"):
        return ARTIFACT_PROTOCOL_CMD
    if t in ("topic", "technique", "tactic"):
        return ARTIFACT_TOPIC
    if t in ("file", "path", "filename"):
        return ARTIFACT_FILE
    return t or "topic"


def _extract_keywords(text: str, min_len: int = 2) -> list[str]:
    """Extract normalized keywords from hypothesis text; deterministic sorted list."""
    if not text or not text.strip():
        return []
    # Lowercase, split on non-alphanumeric, drop short tokens
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        if len(t) >= min_len and t not in seen:
            seen.add(t)
            out.append(t)
    return sorted(out)


def build_queries(
    episode: Episode,
    hypothesis: Optional[Hypothesis] = None,
) -> tuple[list[str], dict[str, Any]]:
    """
    Input: Episode + optional Hypothesis.
    Output: (query_strings, structured_terms) with deterministic ordering.
    - query_strings: list of query strings for KB/OpenCTI.
    - structured_terms: artifact queries by type (ip/domain/hash/protocol_cmd/topic), tags, hypothesis_keywords.
    """
    structured: dict[str, Any] = {
        ARTIFACT_IP: [],
        ARTIFACT_DOMAIN: [],
        ARTIFACT_HASH: [],
        ARTIFACT_PROTOCOL_CMD: [],
        ARTIFACT_TOPIC: [],
        ARTIFACT_FILE: [],
        "tags": [],
        "hypothesis_keywords": [],
    }

    # Artifact queries from episode.artifacts
    for a in episode.artifacts:
        if not isinstance(a, dict):
            continue
        t = _artifact_type(str(a.get("type", "")))
        v = a.get("value")
        if v is None or (isinstance(v, str) and not v.strip()):
            continue
        val = str(v).strip()
        if t in structured and isinstance(structured[t], list):
            if val not in structured[t]:
                structured[t].append(val)
        elif t not in structured:
            structured[t] = [val]

    # Sort artifact lists for determinism
    for k in [ARTIFACT_IP, ARTIFACT_DOMAIN, ARTIFACT_HASH, ARTIFACT_PROTOCOL_CMD, ARTIFACT_TOPIC, ARTIFACT_FILE]:
        if isinstance(structured.get(k), list):
            structured[k] = sorted(set(structured[k]))

    # Tag queries from episode.sequence_tags
    structured["tags"] = sorted(set(episode.sequence_tags)) if episode.sequence_tags else []

    # Hypothesis: keywords from text + suspected_tactics + suspected_intrusion_set
    if hypothesis:
        structured["hypothesis_keywords"] = _extract_keywords(hypothesis.text)
        if hypothesis.suspected_intrusion_set:
            structured["hypothesis_keywords"] = sorted(
                set(structured["hypothesis_keywords"]) | {hypothesis.suspected_intrusion_set.strip()}
            )
        for t in hypothesis.suspected_tactics or []:
            if t and t.strip():
                structured["hypothesis_keywords"] = sorted(
                    set(structured["hypothesis_keywords"]) | {t.strip()}
                )

    # Build flat list of query strings (deterministic order)
    query_strings: list[str] = []
    for typ in [ARTIFACT_IP, ARTIFACT_DOMAIN, ARTIFACT_HASH, ARTIFACT_PROTOCOL_CMD, ARTIFACT_TOPIC, ARTIFACT_FILE]:
        for v in structured.get(typ) or []:
            query_strings.append(v)
    for tag in structured.get("tags") or []:
        query_strings.append(tag)
    for kw in structured.get("hypothesis_keywords") or []:
        query_strings.append(kw)
    query_strings = sorted(set(query_strings))

    return query_strings, structured
