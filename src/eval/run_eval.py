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
from src.eval.metrics import egr_overall, ucr_proxy
from src.pipeline.retrieve_evidence import build_evidence_set
from src.pipeline.run_agents import run_all_agents
from src.pipeline.validate_and_repair import validate_and_repair
from src.pipeline.writeback_pipeline import run_writeback

CSV_COLUMNS = [
    "episode_id", "pass", "evidence_items",
    "citations_triage", "citations_hunt", "citations_response",
    "egr_overall", "ucr_proxy",
    "retrieval_ms", "agents_ms", "validation_ms", "writeback_ms", "total_ms",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_one_episode(
    episode_path: Path,
    root: Path,
) -> tuple[Dict[str, Any], Optional[Dict[str, list[str]]]]:
    """
    Run retrieve (if needed), agents, validate+repair, writeback. Return (metrics_row, errors_by_agent or None).
    """
    episode = Episode.model_validate(json.loads(episode_path.read_text(encoding="utf-8")))
    episode_id = episode.episode_id
    evidence_path = root / "outputs" / "evidence" / f"{episode_id}.json"

    # Retrieval
    t0 = time.perf_counter()
    if not evidence_path.exists():
        build_evidence_set(episode_path)
    evidence_set = EvidenceSet.model_validate(json.loads(evidence_path.read_text(encoding="utf-8")))
    retrieval_ms = int((time.perf_counter() - t0) * 1000)

    # Agents
    t0 = time.perf_counter()
    initial = run_all_agents(episode, evidence_set, trust_signals=None, repair_hint=None, write_outputs=False)
    agents_ms = int((time.perf_counter() - t0) * 1000)

    # Validate + repair
    t0 = time.perf_counter()
    outputs, status = validate_and_repair(episode, evidence_set, initial, trust_signals=None)
    validation_ms = int((time.perf_counter() - t0) * 1000)

    if status != "ok":
        _, errors_by_agent = _validate_all_for_eval(outputs, episode, evidence_set)
        def _cites(aid: str) -> int:
            return len(outputs[aid].citations) if aid in outputs else 0
        row = {
            "episode_id": episode_id,
            "pass": 0,
            "evidence_items": len(evidence_set.items),
            "citations_triage": _cites("triage"),
            "citations_hunt": _cites("hunt_planner"),
            "citations_response": _cites("response_advisor"),
            "egr_overall": round(egr_overall(outputs, evidence_set), 4) if outputs else 0.0,
            "ucr_proxy": round(ucr_proxy(episode, evidence_set, outputs), 4) if outputs else 0.0,
            "retrieval_ms": retrieval_ms,
            "agents_ms": agents_ms,
            "validation_ms": validation_ms,
            "writeback_ms": 0,
            "total_ms": retrieval_ms + agents_ms + validation_ms,
        }
        return row, errors_by_agent

    # Write agent outputs so writeback can load them
    agents_dir = root / "outputs" / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    for aid, out in outputs.items():
        (agents_dir / f"{episode_id}_{aid}.json").write_text(out.model_dump_json(indent=2), encoding="utf-8")

    # Writeback
    t0 = time.perf_counter()
    run_writeback(episode_path, mode="dry_run")
    writeback_ms = int((time.perf_counter() - t0) * 1000)

    total_ms = retrieval_ms + agents_ms + validation_ms + writeback_ms
    egr = egr_overall(outputs, evidence_set)
    ucr = ucr_proxy(episode, evidence_set, outputs)

    row = {
        "episode_id": episode_id,
        "pass": 1,
        "evidence_items": len(evidence_set.items),
        "citations_triage": len(outputs["triage"].citations),
        "citations_hunt": len(outputs["hunt_planner"].citations),
        "citations_response": len(outputs["response_advisor"].citations),
        "egr_overall": round(egr, 4),
        "ucr_proxy": round(ucr, 4),
        "retrieval_ms": retrieval_ms,
        "agents_ms": agents_ms,
        "validation_ms": validation_ms,
        "writeback_ms": writeback_ms,
        "total_ms": total_ms,
    }
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
                "retrieval_ms": 0,
                "agents_ms": 0,
                "validation_ms": 0,
                "writeback_ms": 0,
                "total_ms": 0,
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
        numeric_keys = ["pass", "evidence_items", "citations_triage", "citations_hunt", "citations_response", "egr_overall", "ucr_proxy", "retrieval_ms", "agents_ms", "validation_ms", "writeback_ms", "total_ms"]
        means = {}
        medians = {}
        for k in numeric_keys:
            vals = [r[k] for r in rows if r.get(k) is not None]
            if vals:
                means[k] = round(sum(vals) / len(vals), 4)
                sorted_v = sorted(vals)
                medians[k] = sorted_v[len(sorted_v) // 2] if sorted_v else 0
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
        f.write(f"- Avg total_ms: {summary.get('means', {}).get('total_ms', 0):.0f}\n\n")
        f.write("## Top failures\n\n")
        for fail in failures[:10]:
            f.write(f"- **{fail['episode_id']}**: {fail.get('errors', {})}\n")
        f.write("\n## Common validator errors\n\n")
        from collections import Counter
        err_counts = Counter(all_errors)
        for err, c in err_counts.most_common(10):
            f.write(f"- {c}x: {err[:120]}\n")
    return eval_dir
