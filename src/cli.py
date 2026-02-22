"""
Minimal CLI. validate: load config, write one audit log, print status. No OpenAI/OpenCTI/KB.
"""
import argparse
import sys
import uuid
from pathlib import Path

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
    print("validate ok")
    print(f"  run_id={run_id}")
    print(f"  RUN_MODE={cfg.RUN_MODE}")
    print(f"  PROMPT_VERSION={cfg.PROMPT_VERSION}")
    print(f"  audit: outputs/audit/{run_id}.jsonl")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="python -m src.cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate", help="Load config, write one audit log, print status")
    args = p.parse_args()
    if args.cmd == "validate":
        return cmd_validate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
