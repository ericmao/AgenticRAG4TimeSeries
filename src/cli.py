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
    args = p.parse_args()
    if args.cmd == "validate":
        return cmd_validate()
    if args.cmd == "retrieve":
        return cmd_retrieve(args.episode, args.hypothesis)
    return 0


if __name__ == "__main__":
    sys.exit(main())
