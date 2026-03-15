"""
Optional LLM (Ollama default) analysis for Layer C agents.
Returns a short analysis string given episode, evidence context, and optional time_series_signals.
"""
from __future__ import annotations

from src.contracts.episode import Episode
from src.contracts.evidence import EvidenceSet
from src.agents.common import format_evidence_context


def analyze_episode_with_llm(
    episode: Episode,
    evidence_set: EvidenceSet,
    agent_role: str,
    time_series_signals: dict | None = None,
    max_tokens: int = 500,
) -> str:
    """
    Call LLM (Ollama by default) with evidence + optional time-series context; return reply string.
    On failure (no Ollama, timeout) returns empty string or error message so pipeline can continue.
    """
    try:
        llm = __import__("src.llm.factory", fromlist=["get_llm_for_layer_c"]).get_llm_for_layer_c()
    except Exception as e:
        return f"[LLM unavailable: {e}]"
    evidence_lines = format_evidence_context(evidence_set)
    evidence_text = "\n".join(evidence_lines[:30]) or "(no evidence)"
    entity_id = episode.entity_id or (episode.entities[0] if episode.entities else "unknown")
    ts_block = ""
    if time_series_signals and time_series_signals.get("available"):
        ts_block = (
            "\nTime-series context:\n"
            f"- Trend: {time_series_signals.get('trend_summary', '')}\n"
            f"- Retrieval: {time_series_signals.get('retrieval_summary', '')}\n"
        )
        if time_series_signals.get("markov_anomaly"):
            ma = time_series_signals["markov_anomaly"]
            if isinstance(ma, dict) and "error" not in ma:
                ts_block += f"- Markov anomaly: {ma.get('is_anomaly')} (score: {ma.get('anomaly_score')})\n"
        if time_series_signals.get("bert_anomaly"):
            ba = time_series_signals["bert_anomaly"]
            if isinstance(ba, dict) and "error" not in ba:
                ts_block += f"- BERT anomaly: {ba.get('is_anomaly')} (score: {ba.get('anomaly_score')})\n"
    prompt = (
        f"You are a {agent_role}. Based only on the evidence and optional time-series context below, "
        f"give a brief 2–4 sentence assessment for entity {entity_id}. Do not invent facts. Cite evidence_id when relevant.\n\n"
        f"Evidence:\n{evidence_text}\n"
        f"{ts_block}\n"
        "Brief assessment:"
    )
    try:
        from langchain_core.messages import HumanMessage
        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        text = getattr(response, "content", None) or str(response)
        return (text or "").strip()[:max_tokens]
    except Exception as e:
        return f"[LLM error: {e}]"
