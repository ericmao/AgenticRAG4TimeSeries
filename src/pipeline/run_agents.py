"""
Run all C2 agents on episode + evidence_set. Write validated AgentOutput to outputs/agents/<episode_id>_<agent>.json.
Optional: time_series_signals (USE_TIME_SERIES_SIGNALS), LLM analysis (USE_LANGCHAIN_FOR_ANALYSIS, default Ollama).
Audit-log each agent start/end with latency_ms.
"""
from __future__ import annotations

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


def _merge_trust_with_ts(
    episode: Episode,
    trust_signals: Optional[dict[str, Any]],
    rule_id: Optional[str],
) -> dict[str, Any]:
    from src.config import get_config

    cfg = get_config()
    ts: dict[str, Any] = dict(trust_signals or {})
    if rule_id is not None:
        ts["rule_id"] = rule_id
    if getattr(cfg, "USE_TIME_SERIES_SIGNALS", False):
        try:
            from src.pipeline.time_series_signals import get_time_series_signals

            ts_s = get_time_series_signals(episode)
            ts = {**ts, "time_series_signals": ts_s}
        except Exception:
            ts = {**ts, "time_series_signals": {"available": False, "error": "time_series_signals failed"}}
    return ts


def run_single_agent(
    agent_id: str,
    episode: Episode,
    evidence_set: EvidenceSet,
    trust_signals: Optional[dict[str, Any]] = None,
    repair_hint: Optional[str] = None,
    write_outputs: bool = False,
    *,
    rule_id: Optional[str] = None,
) -> AgentOutput:
    """執行單一代理（供多規則修復或細粒度呼叫）。"""
    root = _repo_root()
    out_dir = root / "outputs" / "agents"
    if write_outputs:
        out_dir.mkdir(parents=True, exist_ok=True)
    run_id = episode.run_id
    episode_id = episode.episode_id

    from src.config import get_config

    cfg = get_config()
    prompt_version = getattr(cfg, "PROMPT_VERSION", "v0.1")

    runners = {
        "triage": run_triage,
        "hunt_planner": run_hunt_planner,
        "response_advisor": run_response_advisor,
    }
    if agent_id not in runners:
        raise ValueError(f"Unknown agent_id: {agent_id}")

    ts = _merge_trust_with_ts(episode, trust_signals, rule_id)
    run_fn = runners[agent_id]

    t0 = time.perf_counter()
    audit_log(
        event={"action": "agent_start", "agent_id": agent_id, "repair_hint": bool(repair_hint), "rule_id": rule_id},
        run_id=run_id,
        episode_id=episode_id,
        component="pipeline.run_agents",
        prompt_version=prompt_version,
    )
    out = run_fn(episode, evidence_set, ts, repair_hint=repair_hint)
    latency_ms = int((time.perf_counter() - t0) * 1000)
    if getattr(cfg, "USE_LANGCHAIN_FOR_ANALYSIS", False):
        try:
            from src.agents.llm_analysis import analyze_episode_with_llm

            ts_signals = ts.get("time_series_signals") if ts else None
            llm_text = analyze_episode_with_llm(
                episode, evidence_set, agent_role=agent_id, time_series_signals=ts_signals
            )
            if llm_text:
                out = out.model_copy(update={"structured": {**out.structured, "llm_analysis": llm_text}})
        except Exception:
            out = out.model_copy(update={"structured": {**out.structured, "llm_analysis": "[LLM analysis skipped]"}})
    AgentOutput.model_validate(out.model_dump())
    if len(out.citations) < 3:
        raise ValueError(f"Agent {agent_id} must have >= 3 citations, got {len(out.citations)}")
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
    return out


def run_agents_for_rule(
    episode: Episode,
    evidence_set: EvidenceSet,
    rule_id: str,
    trust_signals: Optional[dict[str, Any]] = None,
    repair_hint: Optional[str] = None,
    write_outputs: bool = True,
) -> dict[str, AgentOutput]:
    """
    對同一 rule_id 依序跑 triage → hunt_planner → response_advisor。
    trust_signals 會合併 rule_id 與可選 time_series_signals。
    """
    root = _repo_root()
    out_dir = root / "outputs" / "agents"
    if write_outputs:
        out_dir.mkdir(parents=True, exist_ok=True)
    episode_id = episode.episode_id

    outputs: dict[str, AgentOutput] = {}

    for agent_id in ("triage", "hunt_planner", "response_advisor"):
        out = run_single_agent(
            agent_id,
            episode,
            evidence_set,
            trust_signals=trust_signals,
            repair_hint=repair_hint,
            write_outputs=False,
            rule_id=rule_id,
        )
        outputs[agent_id] = out
        if write_outputs:
            path = out_dir / f"{episode_id}_{agent_id}.json"
            path.write_text(out.model_dump_json(indent=2), encoding="utf-8")

    return outputs


def run_all_agents(
    episode: Episode,
    evidence_set: EvidenceSet,
    trust_signals: Optional[dict[str, Any]] = None,
    repair_hint: Optional[str] = None,
    write_outputs: bool = True,
) -> dict[str, AgentOutput]:
    """
    Run triage, hunt_planner, response_advisor（單一「default」規則，與舊行為相容）。
    If write_outputs, write to outputs/agents/<episode_id>_<agent_id>.json.
    """
    return run_agents_for_rule(
        episode,
        evidence_set,
        "default",
        trust_signals=trust_signals,
        repair_hint=repair_hint,
        write_outputs=write_outputs,
    )
