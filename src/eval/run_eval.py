"""
Batch-run episodes, compute metrics, write metrics.csv, summary.json, report.md.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.contracts.agent_output import AgentOutput
from src.contracts.episode import Episode
from src.contracts.evidence import EvidenceSet
from src.agents.hunt_planner import run_hunt_planner
from src.agents.response_advisor import run_response_advisor
from src.agents.triage import run_triage
from src.eval.metrics import egr_overall, ucr_proxy
from src.pipeline.retrieve_evidence import build_evidence_set
from src.pipeline.validate_and_repair import validate_and_repair
from src.pipeline.writeback_pipeline import run_writeback

def _ms_ns(start_ns: int, end_ns: int) -> float:
    """Convert ns delta to ms, 3 decimals, float."""
    return round((end_ns - start_ns) / 1e6, 3)


CSV_COLUMNS = [
    "episode_id", "pass", "evidence_items",
    "citations_triage", "citations_hunt", "citations_response",
    "egr_overall", "ucr_proxy",
    "retrieval_ms", "agents_ms", "validation_ms", "writeback_ms", "total_ms",
    "model_name", "tokens_in", "tokens_out", "cost_usd",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_one_episode(
    episode_path: Path,
    root: Path,
) -> tuple[Dict[str, Any], Optional[Dict[str, list[str]]]]:
    """
    Run full retrieve (always, so IO included), agents (per-agent timing), validate+repair, writeback.
    Latency via perf_counter_ns(), ms kept as float with 3 decimals.
    """
    episode = Episode.model_validate(json.loads(episode_path.read_text(encoding="utf-8")))
    episode_id = episode.episode_id
    evidence_path = root / "outputs" / "evidence" / f"{episode_id}.json"

    # Retrieval: full build_evidence_set (KB load + chunk + score + write evidence file)
    t0 = time.perf_counter_ns()
    build_evidence_set(episode_path)
    evidence_set = EvidenceSet.model_validate(json.loads(evidence_path.read_text(encoding="utf-8")))
    retrieval_ms = _ms_ns(t0, time.perf_counter_ns())

    # Agents: time each agent and sum (include in-memory work; file writes in writeback)
    initial: Dict[str, AgentOutput] = {}
    agents_ms_sum = 0.0
    for agent_id, run_fn in [("triage", run_triage), ("hunt_planner", run_hunt_planner), ("response_advisor", run_response_advisor)]:
        t0 = time.perf_counter_ns()
        initial[agent_id] = run_fn(episode, evidence_set, trust_signals=None, repair_hint=None)
        agents_ms_sum += _ms_ns(t0, time.perf_counter_ns())
    agents_ms = round(agents_ms_sum, 3)

    # Validate + repair
    t0 = time.perf_counter_ns()
    outputs, status = validate_and_repair(episode, evidence_set, initial, trust_signals=None)
    validation_ms = _ms_ns(t0, time.perf_counter_ns())

    # Placeholder for cost/tokens (stub agents: zeros)
    model_name = "stub"
    tokens_in = 0
    tokens_out = 0
    cost_usd = 0.0

    def _cites(aid: str) -> int:
        return len(outputs[aid].citations) if aid in outputs else 0

    def _base_row(ok: bool, wb_ms: float = 0.0) -> Dict[str, Any]:
        total = round(retrieval_ms + agents_ms + validation_ms + wb_ms, 3)
        return {
            "episode_id": episode_id,
            "pass": 1 if ok else 0,
            "evidence_items": len(evidence_set.items),
            "citations_triage": _cites("triage"),
            "citations_hunt": _cites("hunt_planner"),
            "citations_response": _cites("response_advisor"),
            "egr_overall": round(egr_overall(outputs, evidence_set), 4) if outputs else 0.0,
            "ucr_proxy": round(ucr_proxy(episode, evidence_set, outputs), 4) if outputs else 0.0,
            "retrieval_ms": round(retrieval_ms, 3),
            "agents_ms": agents_ms,
            "validation_ms": round(validation_ms, 3),
            "writeback_ms": round(wb_ms, 3),
            "total_ms": total,
            "model_name": model_name,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": cost_usd,
        }

    if status != "ok":
        _, errors_by_agent = _validate_all_for_eval(outputs, episode, evidence_set)
        return _base_row(False), errors_by_agent

    # Write agent outputs (included in writeback timing below via run_writeback's reads, but we need to write first)
    agents_dir = root / "outputs" / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    t0_wb = time.perf_counter_ns()
    for aid, out in outputs.items():
        (agents_dir / f"{episode_id}_{aid}.json").write_text(out.model_dump_json(indent=2), encoding="utf-8")
    run_writeback(episode_path, mode="dry_run")
    writeback_ms = _ms_ns(t0_wb, time.perf_counter_ns())

    row = _base_row(True, writeback_ms)
    row["egr_overall"] = round(egr_overall(outputs, evidence_set), 4)
    row["ucr_proxy"] = round(ucr_proxy(episode, evidence_set, outputs), 4)
    return row, None


def _validate_all_for_eval(
    outputs: Dict[str, AgentOutput],
    episode: Episode,
    evidence_set: EvidenceSet,
) -> tuple[Dict[str, AgentOutput], Dict[str, list[str]]]:
    """Reuse validation logic to get errors_by_agent."""
    from src.validators.citations import validate_citations
    from src.validators.policy_guardrails import validate_response_advisor_policy
    errors_by_agent: Dict[str, list[str]] = {}
    for agent_id, out in outputs.items():
        errs: list[str] = []
        cr = validate_citations(out, evidence_set, min_citations=3)
        if not cr["valid"]:
            errs.extend(cr.get("errors") or [])
        if agent_id == "response_advisor":
            pr = validate_response_advisor_policy(out, episode)
            if not pr["valid"]:
                errs.extend(pr.get("errors") or [])
        if errs:
            errors_by_agent[agent_id] = errs
    return outputs, errors_by_agent


def run_eval(
    episodes_dir: str | Path,
    limit: int = 20,
) -> Path:
    """
    Iterate episode JSONs in episodes_dir (up to limit). For each: retrieve if needed, agents, validate+repair, writeback.
    Append row to outputs/eval/metrics.csv; produce summary.json and report.md.
    Returns path to outputs/eval/.
    """
    root = _repo_root()
    episodes_dir = Path(episodes_dir)
    if not episodes_dir.is_absolute():
        episodes_dir = root / episodes_dir
    if not episodes_dir.is_dir():
        raise FileNotFoundError(f"Episodes dir not found: {episodes_dir}")

    eval_dir = root / "outputs" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    csv_path = eval_dir / "metrics.csv"
    summary_path = eval_dir / "summary.json"
    report_path = eval_dir / "report.md"

    episode_files = sorted(episodes_dir.glob("*.json"))[:limit]
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    all_errors: List[str] = []

    for ep_path in episode_files:
        try:
            row, errors_by_agent = _run_one_episode(ep_path, root)
            rows.append(row)
            if errors_by_agent:
                failures.append({"episode_id": row["episode_id"], "errors": errors_by_agent})
                for errs in errors_by_agent.values():
                    all_errors.extend(errs)
        except Exception as e:
            rows.append({
                "episode_id": ep_path.stem,
                "pass": 0,
                "evidence_items": 0,
                "citations_triage": 0,
                "citations_hunt": 0,
                "citations_response": 0,
                "egr_overall": 0.0,
                "ucr_proxy": 0.0,
                "retrieval_ms": 0.0,
                "agents_ms": 0.0,
                "validation_ms": 0.0,
                "writeback_ms": 0.0,
                "total_ms": 0.0,
                "model_name": "stub",
                "tokens_in": 0,
                "tokens_out": 0,
                "cost_usd": 0.0,
            })
            failures.append({"episode_id": ep_path.stem, "errors": {"exception": [str(e)]}})
            all_errors.append(str(e))

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_COLUMNS})

    # Summary
    n = len(rows)
    if n == 0:
        summary = {"count": 0, "pass_rate": 0.0}
    else:
        numeric_keys = ["pass", "evidence_items", "citations_triage", "citations_hunt", "citations_response", "egr_overall", "ucr_proxy", "retrieval_ms", "agents_ms", "validation_ms", "writeback_ms", "total_ms", "tokens_in", "tokens_out", "cost_usd"]
        means = {}
        medians = {}
        for k in numeric_keys:
            vals = [r[k] for r in rows if r.get(k) is not None]
            if vals:
                means[k] = round(sum(vals) / len(vals), 4)
                sorted_v = sorted(vals)
                medians[k] = sorted_v[len(sorted_v) // 2] if sorted_v else 0
        if any(r.get("model_name") for r in rows):
            means["model_name"] = rows[0].get("model_name", "stub")
        summary = {
            "count": n,
            "pass_rate": round(sum(r.get("pass", 0) for r in rows) / n, 4),
            "means": means,
            "medians": medians,
        }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Eval Report\n\n")
        f.write(f"- Episodes: {n}\n")
        f.write(f"- Pass rate: {summary.get('pass_rate', 0):.2%}\n")
        f.write(f"- Avg EGR: {summary.get('means', {}).get('egr_overall', 0):.4f}\n")
        f.write(f"- Avg UCR_proxy: {summary.get('means', {}).get('ucr_proxy', 0):.4f}\n")
        f.write(f"- Avg total_ms: {summary.get('means', {}).get('total_ms', 0):.3f}\n\n")
        f.write("## Top failures\n\n")
        for fail in failures[:10]:
            f.write(f"- **{fail['episode_id']}**: {fail.get('errors', {})}\n")
        f.write("\n## Common validator errors\n\n")
        from collections import Counter
        err_counts = Counter(all_errors)
        for err, c in err_counts.most_common(10):
            f.write(f"- {c}x: {err[:120]}\n")
    return eval_dir
