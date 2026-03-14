"""
Build EvidenceSet from episode (and optional hypothesis): query build, KB + OpenCTI retrieval, assemble, write, audit.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.contracts.episode import Episode, Hypothesis
from src.contracts.evidence import EvidenceSet
from src.retrievers.assemble import assemble_evidence
from src.retrievers.kb import retrieve_from_kb
from src.retrievers.opencti import retrieve_from_opencti
from src.retrievers.query_builder import build_queries
from src.utils.audit_logger import audit_log


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_evidence_set(
    episode_path: str | Path,
    hypothesis_path: Optional[str | Path] = None,
) -> EvidenceSet:
    """
    Load episode (and optional hypothesis), build queries, retrieve from KB + OpenCTI, assemble, write outputs/evidence/<episode_id>.json.
    Audit-log key steps (queries built, candidate counts, final counts).
    """
    root = _repo_root()
    episode_path = Path(episode_path)
    if not episode_path.is_absolute():
        episode_path = root / episode_path
    episode_data = json.loads(episode_path.read_text(encoding="utf-8"))
    episode = Episode.model_validate(episode_data)
    run_id = episode.run_id
    episode_id = episode.episode_id

    hypothesis: Optional[Hypothesis] = None
    if hypothesis_path:
        hp = Path(hypothesis_path)
        if not hp.is_absolute():
            hp = root / hp
        if hp.exists():
            hypothesis = Hypothesis.model_validate(json.loads(hp.read_text(encoding="utf-8")))

    from src.config import get_config
    cfg = get_config()
    prompt_version = cfg.PROMPT_VERSION

    # Audit: step start
    audit_log(
        event={"action": "retrieve_evidence_start", "episode_id": episode_id},
        run_id=run_id,
        episode_id=episode_id,
        component="pipeline.retrieve_evidence",
        prompt_version=prompt_version,
    )

    # Build queries
    query_strings, structured = build_queries(episode, hypothesis)
    audit_log(
        event={"action": "queries_built", "query_count": len(query_strings), "queries": query_strings[:20]},
        run_id=run_id,
        episode_id=episode_id,
        component="pipeline.retrieve_evidence",
        prompt_version=prompt_version,
    )

    # KB retrieval
    kb_items = retrieve_from_kb(episode, query_strings, hypothesis)
    opencti_items = retrieve_from_opencti(episode, hypothesis, query_strings)
    audit_log(
        event={"action": "candidate_counts", "kb": len(kb_items), "opencti": len(opencti_items)},
        run_id=run_id,
        episode_id=episode_id,
        component="pipeline.retrieve_evidence",
        prompt_version=prompt_version,
    )

    # Assemble
    evidence_set = assemble_evidence(
        kb_items,
        opencti_items,
        episode_id=episode_id,
        run_id=run_id,
        max_items=50,
    )
    audit_log(
        event={
            "action": "evidence_assembled",
            "final_count": len(evidence_set.items),
            "stats": evidence_set.stats,
        },
        run_id=run_id,
        episode_id=episode_id,
        component="pipeline.retrieve_evidence",
        prompt_version=prompt_version,
    )

    # Write outputs/evidence/<episode_id>.json
    out_dir = root / "outputs" / "evidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{episode_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(evidence_set.model_dump_json(indent=2))
    audit_log(
        event={"action": "evidence_written", "path": str(out_path.relative_to(root))},
        run_id=run_id,
        episode_id=episode_id,
        component="pipeline.retrieve_evidence",
        prompt_version=prompt_version,
    )

    return evidence_set
