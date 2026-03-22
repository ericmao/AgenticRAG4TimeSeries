#!/usr/bin/env python3
"""
SenseL EvidenceOps — run CaseOrchestrator on episode + evidence; optional retrieve/writeback/demo report.
Usage:
  PYTHONPATH=. python scripts/run_layerc_case_orchestrator.py --episode tests/demo/episode_insider_highrisk.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main() -> int:
    p = argparse.ArgumentParser(description="SenseL EvidenceOps Case Orchestrator")
    p.add_argument("--episode", required=True, help="Path to episode JSON")
    p.add_argument("--hypothesis", default=None, help="Optional hypothesis JSON path")
    p.add_argument("--skip-retrieve", action="store_true", help="Do not run C1 if evidence missing")
    p.add_argument("--skip-writeback", action="store_true", help="Skip C3 writeback dry_run")
    p.add_argument("--skip-demo-report", action="store_true", help="Skip demo_report.md append")
    args = p.parse_args()

    ep_path = Path(args.episode)
    if not ep_path.is_absolute():
        ep_path = REPO / ep_path
    if not ep_path.is_file():
        print(f"episode not found: {ep_path}", file=sys.stderr)
        return 2

    from src.contracts.episode import Episode, Hypothesis
    from src.contracts.evidence import EvidenceSet
    from src.layer_c.orchestrator.case_orchestrator import CaseOrchestrator
    from src.layer_c.schemas.decision_bundle import HypothesisInput, RiskContextInput
    from src.pipeline.retrieve_evidence import build_evidence_set

    episode = Episode.model_validate(json.loads(ep_path.read_text(encoding="utf-8")))
    episode_id = episode.episode_id
    evidence_path = REPO / "outputs" / "evidence" / f"{episode_id}.json"

    if not args.skip_retrieve and not evidence_path.exists():
        build_evidence_set(ep_path)
    if not evidence_path.exists():
        print(f"evidence not found: {evidence_path}; run without --skip-retrieve or run retrieve first.", file=sys.stderr)
        return 2

    evidence_set = EvidenceSet.model_validate(json.loads(evidence_path.read_text(encoding="utf-8")))

    hyp_in: Optional[HypothesisInput] = None
    if args.hypothesis:
        hp = Path(args.hypothesis)
        if not hp.is_absolute():
            hp = REPO / hp
        hyp_in = HypothesisInput(hypothesis=Hypothesis.model_validate(json.loads(hp.read_text(encoding="utf-8"))))

    risk = RiskContextInput.from_episode(episode)
    orch = CaseOrchestrator(repo_root=REPO)
    state, status = orch.run(episode, evidence_set, risk=risk, hypothesis=hyp_in, write_outputs=True)

    if status != "ok":
        errs = state.routing.get("validation_errors", [])
        print("EvidenceOps validation FAILED", file=sys.stderr)
        for e in errs:
            print(f"  {e}", file=sys.stderr)
        return 1

    from src.layer_c.reports.case_summary_builder import build_case_summary
    from src.layer_c.writeback.decision_bundle_builder import build_and_save_evidenceops_bundle

    build_and_save_evidenceops_bundle(state, episode, evidence_set, REPO, save_legacy_audit=True)
    build_case_summary(state, REPO)

    if not args.skip_writeback:
        from src.layer_c.writeback.writeback_client import run_writeback_dry_run

        run_writeback_dry_run(ep_path, mode="dry_run")

    if not args.skip_demo_report:
        from src.layer_c.reports.demo_report_builder import generate_demo_report_with_evidenceops

        generate_demo_report_with_evidenceops(str(ep_path), state)

    print(f"EvidenceOps ok episode_id={episode_id}")
    print(f"  outputs/evidenceops/decision_bundle_{episode_id}.json")
    print(f"  outputs/evidenceops/case_summary_{episode_id}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
