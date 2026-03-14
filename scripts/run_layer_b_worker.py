#!/usr/bin/env python3
"""
Run Layer B worker: read job configs from folder, run inference, POST to Control Plane.
Usage:
  PYTHONPATH=. python scripts/run_layer_b_worker.py [--jobs-dir jobs] [--no-post]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.layer_b_inference.worker import run_jobs_from_folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs-dir", default="jobs", help="Folder with *.json job configs")
    parser.add_argument("--no-post", action="store_true", help="Do not POST to Control Plane")
    args = parser.parse_args()
    jobs_dir = REPO_ROOT / args.jobs_dir
    results = run_jobs_from_folder(jobs_dir, post_to_control_plane=not args.no_post)
    for job_id, result, fetch_ms in results:
        if result is not None:
            print(f"job={job_id} hypothesis_id={result.hypothesis.hypothesis_id} fetch_ms={fetch_ms:.1f}")
        else:
            print(f"job={job_id} FAILED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
