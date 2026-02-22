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
    """Load episode, load or run retrieve for evidence, run all agents, validate outputs (fail fast)."""
    import json
    from pathlib import Path
    from src.contracts.episode import Episode
    from src.contracts.evidence import EvidenceSet
    from src.pipeline.retrieve_evidence import build_evidence_set
    from src.pipeline.run_agents import run_all_agents

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
    outputs = run_all_agents(episode, evidence_set, trust_signals=None)
    print("analyze ok")
    print(f"  episode_id={episode_id}")
    for aid, out in outputs.items():
        print(f"  {aid}: citations={len(out.citations)} confidence={out.confidence:.2f}")
    print(f"  outputs: outputs/agents/{episode_id}_<agent>.json")
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
    args = p.parse_args()
    if args.cmd == "validate":
        return cmd_validate()
    if args.cmd == "retrieve":
        return cmd_retrieve(args.episode, args.hypothesis)
    if args.cmd == "analyze":
        return cmd_analyze(args.episode, args.hypothesis)
    return 0


if __name__ == "__main__":
    sys.exit(main())
