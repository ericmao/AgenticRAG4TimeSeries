"""
CERT/Episode → KB candidates: read episode JSONs, generate rule-based candidate documents.
Self-evaluation: reliability_score (0-1); when >= reliability_threshold, auto_add=True and can export to KB.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, List, Optional

from src.contracts.episode import Episode
from src.contracts.kb_candidate import KBCandidate

# Tag weights for reliability (high-signal tags contribute more)
TAG_WEIGHTS = {"lateral": 0.4, "burst": 0.3, "device": 0.2, "logon": 0.1, "logoff": 0.1}
TAG_SCORE_CAP = 1.0
EVENT_SCORE_SCALE = 80  # event_count / this, capped 1
HOST_DIVERSITY_CAP = 5  # n_hosts / this, capped 1
WEIGHT_TAG = 0.5
WEIGHT_EVENT = 0.25
WEIGHT_DIVERSITY = 0.25


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _safe_filename(name: str) -> str:
    """Safe filename (alphanumeric, hyphen, underscore only)."""
    return re.sub(r"[^\w\-]", "_", name)


def compute_reliability_score(
    sequence_tags: List[str],
    event_count: int,
    artifacts_summary: dict[str, list[str]],
) -> float:
    """
    Self-evaluation: 0-1 score from tag richness, event volume, host/user diversity.
    """
    tag_score = 0.0
    for t in sequence_tags or []:
        tag_score += TAG_WEIGHTS.get(t.strip().lower(), 0.0)
    tag_score = min(TAG_SCORE_CAP, tag_score) * WEIGHT_TAG

    event_score = min(1.0, event_count / max(1, EVENT_SCORE_SCALE)) * WEIGHT_EVENT

    hosts = artifacts_summary.get("host", [])
    users = artifacts_summary.get("user", [])
    diversity = (len(hosts) + len(users)) / max(1, HOST_DIVERSITY_CAP)
    diversity = min(1.0, diversity) * WEIGHT_DIVERSITY

    raw = tag_score + event_score + diversity
    return round(min(1.0, max(0.0, raw)), 4)


def _artifacts_summary(artifacts: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Group artifact values by type."""
    out: dict[str, list[str]] = {}
    for a in artifacts or []:
        if not isinstance(a, dict):
            continue
        t = str(a.get("type", "")).strip() or "other"
        v = a.get("value")
        if v is None:
            continue
        val = str(v).strip()
        if not val:
            continue
        if t not in out:
            out[t] = []
        if val not in out[t]:
            out[t].append(val)
    for k in out:
        out[k] = sorted(out[k])
    return out


def _generate_body_md(episode: Episode, artifacts_summary: dict[str, list[str]]) -> str:
    """Rule-based Markdown body from episode tags and artifacts."""
    tags = episode.sequence_tags or []
    tags_str = ", ".join(tags) if tags else "none"
    users = artifacts_summary.get("user", [])
    hosts = artifacts_summary.get("host", [])
    user_str = ", ".join(users[:5]) if users else "—"
    if users and len(users) > 5:
        user_str += f" (+{len(users) - 5} more)"
    host_str = ", ".join(hosts[:10]) if hosts else "—"
    if hosts and len(hosts) > 10:
        host_str += f" (+{len(hosts) - 10} more)"
    n = len(episode.events or [])
    lines = [
        f"## Episode: {episode.episode_id}",
        "",
        f"- **Run**: {episode.run_id}",
        f"- **Sequence tags**: {tags_str}",
        f"- **Entities (users)**: {user_str}",
        f"- **Hosts**: {host_str}",
        f"- **Event count**: {n}",
        "",
        "### Pattern summary",
        "",
    ]
    added = False
    if "lateral" in tags:
        lines.append("- Multiple hosts observed in this window (lateral movement signal).")
        added = True
    if "burst" in tags:
        lines.append("- High event count in window (burst signal).")
        added = True
    if "device" in tags:
        lines.append("- Device connect/disconnect activity present.")
        added = True
    if "logon" in tags or "logoff" in tags:
        lines.append("- Logon/logoff activity in window.")
        added = True
    if not added:
        lines.append("- No special tags; baseline activity.")
    lines.append("")
    return "\n".join(lines)


def generate_candidate(
    episode: Episode,
    reliability_threshold: float = 0.0,
) -> KBCandidate:
    """Build one KBCandidate from an Episode (rule-based) and set reliability_score, auto_add."""
    artifacts_summary = _artifacts_summary(episode.artifacts)
    body_md = _generate_body_md(episode, artifacts_summary)
    score = compute_reliability_score(
        episode.sequence_tags or [],
        len(episode.events or []),
        artifacts_summary,
    )
    auto_add = score >= reliability_threshold
    return KBCandidate(
        episode_id=episode.episode_id,
        run_id=episode.run_id,
        sequence_tags=list(episode.sequence_tags or []),
        artifacts_summary=artifacts_summary,
        body_md=body_md,
        event_count=len(episode.events or []),
        source="cert_episode",
        reliability_score=score,
        auto_add=auto_add,
    )


def load_episodes_from_dir(
    episodes_dir: Path,
    limit: Optional[int] = None,
    filter_tags: Optional[List[str]] = None,
) -> List[Episode]:
    """Load episode JSONs from directory; optional limit and filter by sequence_tags (any match)."""
    if not episodes_dir.is_dir():
        return []
    episodes: List[Episode] = []
    for p in sorted(episodes_dir.glob("*.json")):
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            ep = Episode.model_validate(data)
            if filter_tags:
                tags = set(ep.sequence_tags or [])
                if not any(t in tags for t in filter_tags):
                    continue
            episodes.append(ep)
            if limit is not None and len(episodes) >= limit:
                break
        except Exception:
            continue
    return episodes


def run_cert_to_kb_candidates(
    episodes_dir: str | Path,
    out_dir: str | Path = "outputs/kb_candidates/cert",
    filter_tags: Optional[List[str]] = None,
    limit: Optional[int] = None,
    reliability_threshold: float = 0.6,
) -> tuple[List[KBCandidate], Path]:
    """
    Load episodes, generate KBCandidates with reliability_score and auto_add (>= threshold).
    Write to out_dir/candidates.json. Returns (candidates, output_path).
    """
    root = _repo_root()
    episodes_dir = Path(episodes_dir)
    if not episodes_dir.is_absolute():
        episodes_dir = root / episodes_dir
    out_dir = Path(out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = load_episodes_from_dir(episodes_dir, limit=limit, filter_tags=filter_tags)
    candidates: List[KBCandidate] = [
        generate_candidate(ep, reliability_threshold=reliability_threshold)
        for ep in episodes
    ]
    out_path = out_dir / "candidates.json"
    out_path.write_text(
        json.dumps([c.model_dump() for c in candidates], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return candidates, out_path


def export_candidates_to_kb(
    candidates_path: str | Path,
    kb_subdir: str = "cert_candidates",
    reliability_threshold: Optional[float] = None,
    max_export: Optional[int] = 50,
) -> tuple[Path, int]:
    """
    Read candidates.json; export only candidates with reliability_score >= threshold (or auto_add)
    to kb/<kb_subdir>/*.md. Returns (kb_dir, count_written).
    """
    root = _repo_root()
    candidates_path = Path(candidates_path)
    if not candidates_path.is_absolute():
        candidates_path = root / candidates_path
    if not candidates_path.exists():
        return root / "kb" / kb_subdir, 0
    data = json.loads(candidates_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        data = []
    kb_dir = root / "kb" / kb_subdir
    kb_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for raw in data:
        if max_export is not None and written >= max_export:
            break
        try:
            c = KBCandidate.model_validate(raw)
            if reliability_threshold is not None:
                if c.reliability_score < reliability_threshold:
                    continue
            elif not c.auto_add:
                continue
            name = _safe_filename(c.episode_id) + ".md"
            path = kb_dir / name
            path.write_text(c.body_md, encoding="utf-8")
            written += 1
        except Exception:
            continue
    return kb_dir, written
