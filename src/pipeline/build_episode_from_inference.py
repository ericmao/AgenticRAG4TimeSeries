"""
Build Layer C Episode (and optional Hypothesis file) from Layer B InferenceResult + events.
Used to feed Layer C after Layer B has run (e.g. for E2E sample/CERT flow).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

# Allow running from repo root with PYTHONPATH=.
try:
    from contracts import InferenceResult
except ImportError:
    InferenceResult = None  # type: ignore

from src.contracts.episode import Episode
from src.contracts.episode import Hypothesis as LayerCHypothesis


def build_episode_from_inference(
    result: "InferenceResult",
    events: list[dict[str, Any]],
    episode_id: Optional[str] = None,
    run_id: str = "layerc-from-layerb",
) -> tuple[Episode, LayerCHypothesis]:
    """
    Build Episode and Layer C Hypothesis from InferenceResult and events list.
    events: list of event dicts (ts_ms, entity, action, artifact, source, confidence, domain).
    """
    if InferenceResult is None:
        raise RuntimeError("contracts.InferenceResult not available")
    hyp = result.hypothesis
    req_id = result.request_id or result.job_id or result.endpoint_id
    ep_id = episode_id or f"ep-{result.endpoint_id}-{req_id}"[:64]
    t0 = hyp.window.get("start", 0) if hyp.window else 0
    t1 = hyp.window.get("end", 0) if hyp.window else 0
    if not t0 and not t1 and events:
        t0 = min(e.get("ts_ms", 0) for e in events)
        t1 = max(e.get("ts_ms", 0) for e in events)

    entities_set: set[str] = set()
    artifacts_list: list[dict[str, Any]] = []
    for e in events:
        entities_set.add(str(e.get("entity", "")))
        art = e.get("artifact") or {}
        if isinstance(art, dict) and art.get("host"):
            entities_set.add(art["host"])
            artifacts_list.append({"type": "host", "value": art["host"]})
    entities = sorted(entities_set)
    if hyp.entity_id and hyp.entity_id not in entities_set:
        entities = [hyp.entity_id] + [x for x in entities if x != hyp.entity_id]
    # Dedupe artifacts by value
    seen_art: set[tuple[str, str]] = set()
    for a in artifacts_list:
        k = (a.get("type", ""), a.get("value", ""))
        if k not in seen_art:
            seen_art.add(k)
    artifacts = [a for a in artifacts_list if (a.get("type"), a.get("value")) in seen_art]
    if not any(a.get("type") == "user" for a in artifacts) and hyp.entity_id:
        artifacts.insert(0, {"type": "user", "value": hyp.entity_id})

    risk_context: dict[str, Any] = {
        "risk_score": hyp.risk_score,
        "recommendation": hyp.recommendation,
        "ttp_candidates": hyp.ttp_candidates,
        "likelihood": hyp.likelihood,
        "source": "layer_b",
    }

    episode = Episode(
        episode_id=ep_id,
        run_id=run_id,
        t0_ms=t0,
        t1_ms=t1,
        tenant_id=result.tenant_id or None,
        entity_id=hyp.entity_id or result.endpoint_id,
        window=hyp.window or {"start": t0, "end": t1},
        entities=entities,
        artifacts=artifacts[:20],
        sequence_tags=[],
        events=events,
        risk_context=risk_context,
    )

    layer_c_hypothesis = LayerCHypothesis(
        hypothesis_id=hyp.hypothesis_id,
        text=hyp.text or "",
        suspected_intrusion_set=hyp.suspected_intrusion_set,
        suspected_tactics=hyp.suspected_tactics or [t.get("technique_id", t.get("id", "")) for t in hyp.ttp_candidates],
        constraints=hyp.constraints,
    )
    return episode, layer_c_hypothesis


def run_build_episode_from_inference(
    result_path: str | Path,
    events: Optional[list[dict[str, Any]]] = None,
    job_path: Optional[str | Path] = None,
    episode_path: Optional[str | Path] = None,
    hypothesis_path: Optional[str | Path] = None,
    episode_id: Optional[str] = None,
    repo_root: Optional[Path] = None,
) -> tuple[Path, Path]:
    """
    Load InferenceResult from result_path. Events from events list, or from job_path (job JSON with "events" key).
    Write Episode and Hypothesis JSON; return (episode_path, hypothesis_path).
    """
    from contracts import InferenceResult as IR

    path = Path(result_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    result = IR.model_validate(data)
    if events is None and job_path:
        job_data = json.loads(Path(job_path).read_text(encoding="utf-8"))
        events = job_data.get("events", [])
    if events is None:
        events = []
    episode, lc_hyp = build_episode_from_inference(result, events, episode_id=episode_id)
    root = repo_root or path.resolve().parents[1]
    out_dir = root / "outputs" / "episodes_from_b"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ep = Path(episode_path) if episode_path else out_dir / f"{episode.episode_id}.json"
    out_hyp = Path(hypothesis_path) if hypothesis_path else out_dir / f"{episode.episode_id}_hypothesis.json"
    out_ep.write_text(episode.model_dump_json(indent=2), encoding="utf-8")
    out_hyp.write_text(lc_hyp.model_dump_json(indent=2), encoding="utf-8")
    return out_ep, out_hyp
