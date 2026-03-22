"""
Layer C investigation graph: map analysis_runs rows to LayerCCasePayload JSON for the SPA.
"""
from __future__ import annotations

from typing import Any, Optional


def _triage_level_from_agent_outputs(ao: Any) -> str:
    if not isinstance(ao, dict):
        return "—"
    br = ao.get("by_rule")
    if isinstance(br, dict) and br:
        first = next(iter(br.values()))
        if isinstance(first, dict):
            tri = first.get("triage")
            if isinstance(tri, dict):
                st = tri.get("structured")
                if isinstance(st, dict) and st.get("triage_level"):
                    return str(st["triage_level"])
    tri = ao.get("triage")
    if isinstance(tri, dict):
        st = tri.get("structured")
        if isinstance(st, dict) and st.get("triage_level"):
            return str(st["triage_level"])
    return "—"


def _top_action_from_agent_outputs(ao: Any) -> str:
    if not isinstance(ao, dict):
        return "—"
    ra = ao.get("response_advisor")
    if not isinstance(ra, dict):
        br = ao.get("by_rule")
        if isinstance(br, dict) and br:
            first = next(iter(br.values()))
            if isinstance(first, dict):
                ra = first.get("response_advisor")
    if not isinstance(ra, dict):
        return "—"
    st = ra.get("structured")
    if not isinstance(st, dict):
        return "—"
    actions = st.get("actions")
    if not isinstance(actions, list) or not actions:
        return "—"
    a0 = actions[0]
    if isinstance(a0, dict) and a0.get("action"):
        return str(a0["action"])
    return "—"


def run_row_to_layerc_payload(row: dict[str, Any]) -> dict[str, Any]:
    """Shape expected by @agentic/layerc-graph-ui LayerCCasePayload."""
    ev = row.get("evidence_json")
    if not isinstance(ev, dict):
        ev = {}
    episode_id = row.get("episode_id") or ev.get("episode_id") or ""
    target_ip = row.get("target_ip")
    entities: list[str] = []
    if target_ip:
        entities.append(str(target_ip))
    episode: dict[str, Any] = {
        "episode_id": episode_id,
        "run_id": row.get("run_id"),
        "t0_ms": row.get("window_start_ms"),
        "t1_ms": row.get("window_end_ms"),
        "entities": entities,
        "artifacts": [],
    }
    return {
        "episode_id": episode_id,
        "run_id": row.get("run_id"),
        "status": row.get("status"),
        "target_ip": target_ip,
        "alert_count": row.get("alert_count"),
        "episode": episode,
        "evidence_json": ev,
        "agent_outputs_json": row.get("agent_outputs_json") or {},
        "writeback_json": row.get("writeback_json"),
    }


def _triage_from_list_row(row: dict[str, Any]) -> str:
    ls = row.get("layerc_summary")
    if isinstance(ls, dict):
        tl = ls.get("triage_level")
        if tl:
            return str(tl)
    return "—"


def run_row_to_episode_list_entry(row: dict[str, Any]) -> dict[str, Any]:
    """list_runs 列不含 agent_outputs_json；用 layerc_summary 顯示 triage。"""
    ao = row.get("agent_outputs_json")
    triage = _triage_from_list_row(row)
    if triage == "—" and ao is not None:
        triage = _triage_level_from_agent_outputs(ao)
    top_act = "—"
    if ao is not None:
        top_act = _top_action_from_agent_outputs(ao)
    rid = row.get("id")
    return {
        "episode_id": row.get("episode_id") or "",
        "run_id": int(rid) if rid is not None else None,
        "triage_level": triage,
        "alert_count": int(row.get("alert_count") or 0),
        "top_entity": str(row.get("target_ip") or "—"),
        "top_action": top_act,
        "status": str(row.get("status") or ""),
        "source": str(row.get("source") or ""),
    }
