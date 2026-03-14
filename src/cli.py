"""
Minimal CLI. validate: load config, write one audit log, print status. No OpenAI/OpenCTI/KB.
"""
import argparse
import sys
import uuid
from pathlib import Path
from typing import Optional

# Ensure repo root on path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import get_config
from src.utils.audit_logger import audit_log


def cmd_validate() -> int:
    cfg = get_config()
    run_id = uuid.uuid4().hex[:12]
    audit_log(
        event={"action": "validate", "run_mode": cfg.RUN_MODE},
        run_id=run_id,
        episode_id=None,
        component="cli",
        prompt_version=cfg.PROMPT_VERSION,
    )
    # Export Layer C JSON schemas and validate sample episode
    from src.contracts.export_schema import run as export_schemas_run
    export_schemas_run()
    print("validate ok")
    print(f"  run_id={run_id}")
    print(f"  RUN_MODE={cfg.RUN_MODE}")
    print(f"  PROMPT_VERSION={cfg.PROMPT_VERSION}")
    print(f"  audit: outputs/audit/{run_id}.jsonl")
    return 0


def cmd_analyze(episode_path: str, hypothesis_path: Optional[str] = None) -> int:
    """Ensure evidence exists, run agents, validate+repair; write outputs/agents/ only when validated."""
    import json
    from pathlib import Path
    from src.contracts.episode import Episode
    from src.contracts.evidence import EvidenceSet
    from src.pipeline.retrieve_evidence import build_evidence_set
    from src.pipeline.validate_and_repair import run_analyze_with_validation

    root = _REPO_ROOT
    ep_path = Path(episode_path)
    if not ep_path.is_absolute():
        ep_path = root / ep_path
    episode = Episode.model_validate(json.loads(ep_path.read_text(encoding="utf-8")))
    episode_id = episode.episode_id
    evidence_path = root / "outputs" / "evidence" / f"{episode_id}.json"
    if not evidence_path.exists():
        build_evidence_set(episode_path, hypothesis_path)
    evidence_set = EvidenceSet.model_validate(json.loads(evidence_path.read_text(encoding="utf-8")))
    outputs, status = run_analyze_with_validation(episode, evidence_set, trust_signals=None)
    if status == "ok":
        print("analyze ok")
        print(f"  episode_id={episode_id}")
        for aid, out in outputs.items():
            print(f"  {aid}: citations={len(out.citations)} confidence={out.confidence:.2f}")
        print(f"  outputs: outputs/agents/{episode_id}_<agent>.json")
        return 0
    print("analyze failed (validation)")
    print(f"  episode_id={episode_id}")
    print(f"  issues: outputs/issues/{episode_id}.json")
    return 1


def cmd_writeback(episode_path: str, mode: str = "dry_run") -> int:
    """Load episode, evidence, agent outputs; build writeback patch (dry_run); write outputs/writeback + decision bundle."""
    from src.pipeline.writeback_pipeline import run_writeback
    patch, writeback_path = run_writeback(episode_path, mode=mode)
    root = _REPO_ROOT
    print("writeback ok")
    print(f"  episode_id={patch.episode_id}")
    print(f"  mode={patch.mode}")
    print(f"  stats={patch.stats}")
    print(f"  output: {writeback_path.relative_to(root)}")
    print(f"  decision_bundle: outputs/audit/decision_bundle_{patch.episode_id}.json")
    return 0


def cmd_eval(episodes_dir: str, limit: int = 20) -> int:
    """Batch-run episodes, write metrics.csv, summary.json, report.md to outputs/eval/."""
    from src.eval.run_eval import run_eval
    eval_dir = run_eval(episodes_dir, limit=limit)
    print("eval ok")
    print(f"  episodes_dir={episodes_dir}")
    print(f"  limit={limit}")
    print(f"  outputs: {eval_dir / 'metrics.csv'}, {eval_dir / 'summary.json'}, {eval_dir / 'report.md'}")
    return 0


def cmd_demo_report(episode_path: str) -> int:
    """Generate outputs/demo/demo_report.md from episode and optional outputs artifacts."""
    from src.pipeline.demo_report import generate_demo_report
    report_path = generate_demo_report(episode_path)
    root = _REPO_ROOT
    try:
        rel = report_path.relative_to(root)
    except ValueError:
        rel = report_path
    print("demo_report ok")
    print(f"  report: {rel}")
    return 0


def cmd_cert2episodes(
    data_dir: str = "data",
    out_dir: str = "outputs/episodes/cert",
    window_days: int = 7,
    run_id: str = "cert-run-1",
) -> int:
    """Build Episode JSONs from CERT (or synthetic) data; write to out_dir."""
    from src.pipeline.cert_to_episodes import run_cert_to_episodes

    episodes, out_path, count_written = run_cert_to_episodes(
        data_dir=data_dir,
        out_dir=out_dir,
        window_days=window_days,
        run_id=run_id,
    )
    print("cert2episodes ok")
    print(f"  count_written={count_written}")
    print(f"  path={out_path}")
    first_3 = [ep.episode_id for ep in episodes[:3]]
    print(f"  first_3_episode_ids={first_3}")
    return 0


def cmd_retrieve(episode_path: str, hypothesis_path: Optional[str] = None) -> int:
    """Run C1 retrieval: build EvidenceSet from episode (and optional hypothesis), write outputs/evidence/<episode_id>.json."""
    from src.pipeline.retrieve_evidence import build_evidence_set
    evidence_set = build_evidence_set(episode_path, hypothesis_path)
    print("retrieve ok")
    print(f"  episode_id={evidence_set.episode_id}")
    print(f"  run_id={evidence_set.run_id}")
    print(f"  items={len(evidence_set.items)} (max_items={evidence_set.max_items})")
    print(f"  output: outputs/evidence/{evidence_set.episode_id}.json")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="python -m src.cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate", help="Load config, write one audit log, print status")
    ret_p = sub.add_parser("retrieve", help="KB-first retrieval → EvidenceSet (offline with KB only)")
    ret_p.add_argument("--episode", required=True, help="Path to episode JSON")
    ret_p.add_argument("--hypothesis", default=None, help="Optional path to hypothesis JSON")
    ana_p = sub.add_parser("analyze", help="Run C2 agents on episode (+ evidence; run retrieve if needed)")
    ana_p.add_argument("--episode", required=True, help="Path to episode JSON")
    ana_p.add_argument("--hypothesis", default=None, help="Optional path to hypothesis JSON")
    wb_p = sub.add_parser("writeback", help="C3 writeback: build patch (dry_run), write outputs/writeback + decision bundle")
    wb_p.add_argument("--episode", required=True, help="Path to episode JSON")
    wb_p.add_argument("--mode", default="dry_run", choices=["dry_run", "review", "auto"], help="Writeback mode")
    ev_p = sub.add_parser("eval", help="Batch eval: run episodes, write metrics.csv, summary.json, report.md")
    ev_p.add_argument("--episodes_dir", default="tests/samples/episodes", help="Directory of episode JSONs")
    ev_p.add_argument("--limit", type=int, default=20, help="Max episodes to run")
    c2e_p = sub.add_parser("cert2episodes", help="CERT → Episodes: build episode JSONs from logon/device (or synthetic)")
    c2e_p.add_argument("--data_dir", default="data", help="CERT data directory (logon.csv, device.csv)")
    c2e_p.add_argument("--out_dir", default="outputs/episodes/cert", help="Output directory for episode JSONs")
    c2e_p.add_argument("--window_days", type=int, default=7, help="Time window in days per episode")
    c2e_p.add_argument("--run_id", default="cert-run-1", help="Run ID for generated episodes")
    demo_p = sub.add_parser("demo_report", help="Generate end-to-end demo report (outputs/demo/demo_report.md)")
    demo_p.add_argument("--episode", required=True, help="Path to episode JSON")
    args = p.parse_args()
    if args.cmd == "validate":
        return cmd_validate()
    if args.cmd == "retrieve":
        return cmd_retrieve(args.episode, args.hypothesis)
    if args.cmd == "analyze":
        return cmd_analyze(args.episode, args.hypothesis)
    if args.cmd == "writeback":
        return cmd_writeback(args.episode, mode=getattr(args, "mode", "dry_run"))
    if args.cmd == "eval":
        return cmd_eval(getattr(args, "episodes_dir", "tests/samples/episodes"), getattr(args, "limit", 20))
    if args.cmd == "cert2episodes":
        return cmd_cert2episodes(
            data_dir=getattr(args, "data_dir", "data"),
            out_dir=getattr(args, "out_dir", "outputs/episodes/cert"),
            window_days=getattr(args, "window_days", 7),
            run_id=getattr(args, "run_id", "cert-run-1"),
        )
    if args.cmd == "demo_report":
        return cmd_demo_report(args.episode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
