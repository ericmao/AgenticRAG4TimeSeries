#!/usr/bin/env python3
"""
E2E: Sample/CERT data -> Layer B job -> Layer B inference -> Episode + Hypothesis -> Layer C (retrieve, analyze, writeback).
Uses sample episode JSON (e.g. tests/demo/episode_insider_highrisk.json) or CERT data to drive the full pipeline.
Usage:
  PYTHONPATH=. python scripts/run_e2e_sample_cert.py --episode tests/demo/episode_insider_highrisk.json
  PYTHONPATH=. python scripts/run_e2e_sample_cert.py --cert-data data --window-days 7
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="E2E: Sample/CERT -> Layer B -> Layer C")
    parser.add_argument("--episode", type=str, help="Path to episode JSON (used as source for Layer B job)")
    parser.add_argument("--cert-data", type=str, default="data", help="If no --episode, run cert2episodes from this dir")
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--out-dir", type=str, default="outputs/e2e_sample_cert", help="Output dir for artifacts")
    parser.add_argument("--dry-run", action="store_true", help="Only print steps, do not run")
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve episode source
    episode_path: Path | None = None
    if args.episode:
        episode_path = REPO_ROOT / args.episode
        if not episode_path.exists():
            print(f"Episode file not found: {episode_path}", file=sys.stderr)
            return 1
    else:
        # Run cert2episodes and take first episode
        print("Running cert2episodes...")
        r = subprocess.run(
            [sys.executable, "-m", "src.cli", "cert2episodes", "--data_dir", args.cert_data, "--out_dir", "outputs/episodes/cert"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print("cert2episodes failed:", r.stderr or r.stdout, file=sys.stderr)
            return 1
        episodes_dir = REPO_ROOT / "outputs" / "episodes" / "cert"
        if not episodes_dir.exists():
            print("No episodes dir after cert2episodes", file=sys.stderr)
            return 1
        eps = list(episodes_dir.glob("*.json"))
        if not eps:
            print("No episode JSONs found", file=sys.stderr)
            return 1
        episode_path = eps[0]
        print(f"Using first episode: {episode_path.name}")

    ep_data = json.loads(episode_path.read_text(encoding="utf-8"))
    episode_id = ep_data.get("episode_id", "ep-unknown")
    events = ep_data.get("events", [])
    t0_ms = ep_data.get("t0_ms", 0)
    t1_ms = ep_data.get("t1_ms", 0)
    entities = ep_data.get("entities", [])
    entity_primary = entities[0] if entities else "unknown"

    # 1) Write Layer B job
    job_data = {
        "request": {
            "request_id": f"req-{episode_id}",
            "job_id": f"job-{episode_id}",
            "tenant_id": ep_data.get("tenant_id", "tenant-default"),
            "endpoint_id": entity_primary,
            "t0_ms": t0_ms,
            "t1_ms": t1_ms,
        },
        "events": events,
    }
    job_path = out_dir / "layerb_job.json"
    job_path.write_text(json.dumps(job_data, indent=2), encoding="utf-8")
    print(f"Wrote Layer B job: {job_path}")

    if args.dry_run:
        print("Dry-run: would run Layer B, build episode, run Layer C")
        return 0

    # 2) Run Layer B
    print("Running Layer B inference...")
    r = subprocess.run(
        [sys.executable, "scripts/run_layerb_job.py", "--job", str(job_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    if r.returncode != 0:
        print("Layer B failed:", r.stderr or r.stdout, file=sys.stderr)
        return 1
    result_data = json.loads(r.stdout)
    result_path = out_dir / "layerb_result.json"
    result_path.write_text(json.dumps(result_data, indent=2), encoding="utf-8")
    print(f"Layer B result: {result_path} (risk_score={result_data.get('hypothesis', {}).get('risk_score')})")

    # 3) Build Episode + Hypothesis from result
    from src.pipeline.build_episode_from_inference import run_build_episode_from_inference

    ep_out, hyp_out = run_build_episode_from_inference(
        result_path,
        job_path=job_path,
        episode_path=out_dir / f"{episode_id}.json",
        hypothesis_path=out_dir / f"{episode_id}_hypothesis.json",
        episode_id=episode_id,
        repo_root=REPO_ROOT,
    )
    print(f"Episode from B: {ep_out}")
    print(f"Hypothesis for Layer C: {hyp_out}")

    # 4) Layer C retrieve
    print("Layer C: retrieve...")
    r = subprocess.run(
        [sys.executable, "-m", "src.cli", "retrieve", "--episode", str(ep_out), "--hypothesis", str(hyp_out)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print("Layer C retrieve failed:", r.stderr or r.stdout, file=sys.stderr)
        return 1
    print(r.stdout or "retrieve ok")

    # 5) Layer C analyze
    print("Layer C: analyze...")
    r = subprocess.run(
        [sys.executable, "-m", "src.cli", "analyze", "--episode", str(ep_out), "--hypothesis", str(hyp_out)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print("Layer C analyze failed:", r.stderr or r.stdout, file=sys.stderr)
        return 1
    print(r.stdout or "analyze ok")

    # 6) Layer C writeback (dry_run)
    print("Layer C: writeback (dry_run)...")
    r = subprocess.run(
        [sys.executable, "-m", "src.cli", "writeback", "--episode", str(ep_out), "--mode", "dry_run"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print("Layer C writeback failed:", r.stderr or r.stdout, file=sys.stderr)
        return 1
    print(r.stdout or "writeback ok")

    print("\nE2E completed successfully.")
    print(f"  Episode: {ep_out}")
    print(f"  Evidence: outputs/evidence/{episode_id}.json")
    print(f"  Agents: outputs/agents/{episode_id}_*.json")
    print(f"  Writeback: outputs/writeback/{episode_id}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
