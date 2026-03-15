"""
Time-series signals for Layer C: optional context from time_series_retrieval, trend, Markov/BERT anomaly.
When USE_TIME_SERIES_SIGNALS is true, run_agents calls get_time_series_signals(episode) and passes
result in trust_signals["time_series_signals"]. When CERT/index not available, returns available: false.
"""
from __future__ import annotations

from typing import Any, Optional

from src.contracts.episode import Episode


def get_time_series_signals(
    episode: Episode,
    data_processor: Any = None,
    vector_store: Any = None,
    markov_detector: Any = None,
    bert_detector: Any = None,
) -> dict[str, Any]:
    """
    Build time-series context for this episode (entity_id as user_id).
    Returns dict with available, entity_id, trend_summary, retrieval_summary, markov_anomaly, bert_anomaly, error.
    If dependencies are missing or entity not in index, available is False and pipeline continues without.
    """
    entity_id = episode.entity_id or (episode.entities[0] if episode.entities else None)
    out: dict[str, Any] = {
        "available": False,
        "entity_id": entity_id or "",
        "trend_summary": "",
        "retrieval_summary": "",
        "markov_anomaly": None,
        "bert_anomaly": None,
        "error": None,
    }
    if not entity_id:
        out["error"] = "no entity_id"
        return out

    try:
        # Optional: try to get components from cache/singleton if not passed
        if data_processor is None and vector_store is None:
            # Don't load heavy CERT pipeline by default; caller can inject
            return out

        parts: list[str] = []

        # Trend (if data_processor has user_features)
        if data_processor is not None and getattr(data_processor, "user_features", None) is not None:
            try:
                import re
                user_match = re.search(r"USER\d+", str(entity_id))
                uid = user_match.group() if user_match else entity_id
                uf = data_processor.user_features
                if hasattr(uf, "columns") and "user" in uf.columns and uid in uf["user"].astype(str).values:
                    user_data = uf[uf["user"].astype(str) == uid].sort_values("date")
                    if not user_data.empty and "total_activities" in user_data.columns:
                        rolling = user_data["total_activities"].rolling(window=7, min_periods=1).mean()
                        trend = (rolling.iloc[-1] - rolling.iloc[0]) / max(len(rolling), 1)
                        direction = "increasing" if trend > 0 else "decreasing" if trend < 0 else "stable"
                        parts.append(f"Trend (7d): {direction}; activities rolling mean {float(rolling.iloc[-1]):.1f}.")
            except Exception as e:
                parts.append(f"Trend error: {e}")
        if parts:
            out["trend_summary"] = " ".join(parts)

        # Retrieval (similar sequences)
        if vector_store is not None:
            try:
                results = vector_store.search_similar_sequences(entity_id, 3)
                if results:
                    out["retrieval_summary"] = f"Found {len(results)} similar sequences (top similarity: {results[0].get('similarity_score', 0):.2f})."
            except Exception as e:
                out["retrieval_summary"] = f"Retrieval error: {e}"

        # Markov anomaly
        if markov_detector is not None and getattr(markov_detector, "is_fitted", False) and vector_store is not None:
            try:
                user_embeddings = vector_store.get_user_sequence_embeddings(entity_id)
                if user_embeddings:
                    results = markov_detector.detect_anomalies({entity_id: user_embeddings})
                    if entity_id in results:
                        r = results[entity_id]
                        out["markov_anomaly"] = {"is_anomaly": r.get("is_anomaly"), "anomaly_score": r.get("anomaly_score"), "explanation": r.get("explanation", "")}
            except Exception as e:
                out["markov_anomaly"] = {"error": str(e)}

        # BERT anomaly
        if bert_detector is not None and getattr(bert_detector, "is_fitted", False) and data_processor is not None:
            try:
                bert_seqs = getattr(data_processor, "prepare_bert_sequences", lambda: {})()
                if isinstance(bert_seqs, dict) and entity_id in bert_seqs:
                    user_emb = bert_detector.extract_bert_features({entity_id: bert_seqs[entity_id]})
                    if entity_id in user_emb:
                        results = bert_detector.detect_anomalies(user_emb)
                        if entity_id in results:
                            r = results[entity_id]
                            out["bert_anomaly"] = {"is_anomaly": r.get("is_anomaly"), "anomaly_score": r.get("anomaly_score"), "explanation": r.get("explanation", "")}
            except Exception as e:
                out["bert_anomaly"] = {"error": str(e)}

        out["available"] = bool(out["trend_summary"] or out["retrieval_summary"] or out["markov_anomaly"] or out["bert_anomaly"])
    except Exception as e:
        out["error"] = str(e)
    return out
