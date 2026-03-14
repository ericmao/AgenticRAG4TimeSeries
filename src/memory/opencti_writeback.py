"""
Build writeback patch from episode, evidence_set, agent_outputs. Apply in dry_run (log only) or auto (stub).
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

from src.contracts.agent_output import AgentOutput
from src.contracts.episode import Episode
from src.contracts.evidence import EvidenceSet
from src.contracts.writeback import WritebackPatch


def _provenance(evidence_ids: list[str], source: str = "pipeline") -> dict[str, Any]:
    return {
        "evidence_ids": evidence_ids,
        "generated_at_ms": int(time.time() * 1000),
        "source": source,
    }


def build_writeback_patch(
    episode: Episode,
    evidence_set: EvidenceSet,
    agent_outputs: Dict[str, AgentOutput],
    mode: str = "dry_run",
) -> WritebackPatch:
    """
    Extract stix_id from evidence where source==opencti (may be none for KB-only).
    If no stix_id: notes[] with triage/hunt/response summaries + citations + episode artifacts; stats no_opencti_objects.
    If stix_id exists: sightings (stix indicator -> artifacts), relationships (entities -> stix_id).
    Deterministic ordering (sort by id strings).
    """
    generated_at_ms = int(time.time() * 1000)
    opencti_items = [i for i in evidence_set.items if i.source == "opencti" and i.stix_id]
    stix_ids = sorted(set(i.stix_id for i in opencti_items if i.stix_id))

    sightings: List[dict[str, Any]] = []
    relationships: List[dict[str, Any]] = []
    notes: List[dict[str, Any]] = []

    if not stix_ids:
        # KB-only: notes with agent summaries + citations + episode artifacts
        for agent_id in sorted(agent_outputs.keys()):
            out = agent_outputs[agent_id]
            note = {
                "type": "agent_summary",
                "agent_id": agent_id,
                "summary": out.summary,
                "citations": list(out.citations),
                "episode_artifacts": list(episode.artifacts or []),
                "provenance": _provenance(out.citations, "pipeline"),
            }
            note["provenance"]["generated_at_ms"] = generated_at_ms
            notes.append(note)
        stats = {"no_opencti_objects": True, "notes_count": len(notes)}
    else:
        # Best-effort: sightings linking stix indicator to episode artifacts; relationships entities -> stix_id
        for stix_id in stix_ids:
            ev_ids = sorted(set(i.evidence_id for i in opencti_items if i.stix_id == stix_id))
            for a in sorted(episode.artifacts or [], key=lambda x: str(x.get("value", ""))):
                if not isinstance(a, dict):
                    continue
                sightings.append({
                    "stix_id": stix_id,
                    "artifact_type": a.get("type"),
                    "artifact_value": a.get("value"),
                    "provenance": _provenance(ev_ids, "opencti"),
                })
            for entity in sorted(episode.entities or []):
                relationships.append({
                    "from_entity": entity,
                    "to_stix_id": stix_id,
                    "relationship_type": "related-to",
                    "provenance": _provenance(ev_ids, "opencti"),
                })
        for agent_id in sorted(agent_outputs.keys()):
            out = agent_outputs[agent_id]
            notes.append({
                "type": "agent_summary",
                "agent_id": agent_id,
                "summary": out.summary,
                "citations": list(out.citations),
                "provenance": _provenance(out.citations, "pipeline"),
            })
        for n in notes:
            n["provenance"]["generated_at_ms"] = generated_at_ms
        stats = {"opencti_objects": len(stix_ids), "sightings_count": len(sightings), "relationships_count": len(relationships), "notes_count": len(notes)}

    # Deterministic sort notes by agent_id
    notes.sort(key=lambda x: (x.get("agent_id", ""), x.get("type", "")))
    sightings.sort(key=lambda x: (x.get("stix_id", ""), str(x.get("artifact_value", ""))))
    relationships.sort(key=lambda x: (x.get("from_entity", ""), x.get("to_stix_id", "")))

    return WritebackPatch(
        episode_id=episode.episode_id,
        run_id=episode.run_id,
        mode=mode,
        sightings=sightings,
        relationships=relationships,
        notes=notes,
        stats=stats,
    )


def apply_patch(patch: WritebackPatch) -> None:
    """
    dry_run: do nothing (only log).
    review: same as dry_run (log only).
    auto: stub — raise NotImplementedError unless OPENCTI_URL/TOKEN configured.
    """
    if patch.mode in ("dry_run", "review"):
        return
    if patch.mode == "auto":
        from src.config import get_config
        cfg = get_config()
        if not cfg.OPENCTI_URL or not cfg.OPENCTI_TOKEN:
            raise NotImplementedError("apply_patch(mode=auto) requires OPENCTI_URL and OPENCTI_TOKEN")
        raise NotImplementedError("apply_patch(mode=auto) not implemented; use dry_run")
