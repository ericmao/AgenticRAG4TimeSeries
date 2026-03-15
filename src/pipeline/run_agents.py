"""
Run all C2 agents on episode + evidence_set. Write validated AgentOutput to outputs/agents/<episode_id>_<agent>.json.
Optional: time_series_signals (USE_TIME_SERIES_SIGNALS), LLM analysis (USE_LANGCHAIN_FOR_ANALYSIS, default Ollama).
Audit-log each agent start/end with latency_ms.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from src.contracts.agent_output import AgentOutput
from src.contracts.episode import Episode
from src.contracts.evidence import EvidenceSet
from src.agents.triage import run_triage
from src.agents.hunt_planner import run_hunt_planner
from src.agents.response_advisor import run_response_advisor
from src.utils.audit_logger import audit_log


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_all_agents(
    episode: Episode,
    evidence_set: EvidenceSet,
    trust_signals: Optional[dict[str, Any]] = None,
    repair_hint: Optional[str] = None,
    write_outputs: bool = True,
) -> dict[str, AgentOutput]:
    """
    Run triage, hunt_planner, response_advisor. If write_outputs, write to outputs/agents/<episode_id>_<agent_id>.json.
    When USE_TIME_SERIES_SIGNALS: merge time_series_signals into trust_signals.
    When USE_LANGCHAIN_FOR_ANALYSIS: call LLM (Ollama default) and add structured["llm_analysis"] to each output.
    Audit-log start/end with latency_ms. Returns dict agent_id -> AgentOutput.
    """
    root = _repo_root()
    out_dir = root / "outputs" / "agents"
    if write_outputs:
        out_dir.mkdir(parents=True, exist_ok=True)
    run_id = episode.run_id
    episode_id = episode.episode_id

    from src.config import get_config
    cfg = get_config()
    prompt_version = getattr(cfg, "PROMPT_VERSION", "v0.1")

    # Optional: time-series signals for trust_signals
    if trust_signals is None:
        trust_signals = {}
    if getattr(cfg, "USE_TIME_SERIES_SIGNALS", False):
        try:
            from src.pipeline.time_series_signals import get_time_series_signals
            ts_signals = get_time_series_signals(episode)
            trust_signals = {**trust_signals, "time_series_signals": ts_signals}
        except Exception:
            trust_signals = {**trust_signals, "time_series_signals": {"available": False, "error": "time_series_signals failed"}}

    runners = [
        ("triage", run_triage),
        ("hunt_planner", run_hunt_planner),
        ("response_advisor", run_response_advisor),
    ]
    outputs: dict[str, AgentOutput] = {}

    for agent_id, run_fn in runners:
        t0 = time.perf_counter()
        audit_log(
            event={"action": "agent_start", "agent_id": agent_id, "repair_hint": bool(repair_hint)},
            run_id=run_id,
            episode_id=episode_id,
            component="pipeline.run_agents",
            prompt_version=prompt_version,
        )
        out = run_fn(episode, evidence_set, trust_signals, repair_hint=repair_hint)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        # Optional: LLM (Ollama) analysis appended to structured
        if getattr(cfg, "USE_LANGCHAIN_FOR_ANALYSIS", False):
            try:
                from src.agents.llm_analysis import analyze_episode_with_llm
                ts_signals = trust_signals.get("time_series_signals") if trust_signals else None
                llm_text = analyze_episode_with_llm(episode, evidence_set, agent_role=agent_id, time_series_signals=ts_signals)
                if llm_text:
                    out = out.model_copy(update={"structured": {**out.structured, "llm_analysis": llm_text}})
            except Exception:
                out = out.model_copy(update={"structured": {**out.structured, "llm_analysis": "[LLM analysis skipped]"}})
        AgentOutput.model_validate(out.model_dump())
        if len(out.citations) < 3:
            raise ValueError(f"Agent {agent_id} must have >= 3 citations, got {len(out.citations)}")
        outputs[agent_id] = out
        audit_log(
            event={"action": "agent_end", "agent_id": agent_id, "latency_ms": latency_ms, "citations": len(out.citations)},
            run_id=run_id,
            episode_id=episode_id,
            component="pipeline.run_agents",
            prompt_version=prompt_version,
        )
        if write_outputs:
            path = out_dir / f"{episode_id}_{agent_id}.json"
            path.write_text(out.model_dump_json(indent=2), encoding="utf-8")

    return outputs
