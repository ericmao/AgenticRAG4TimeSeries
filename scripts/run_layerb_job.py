#!/usr/bin/env python3
"""
Run a single Layer B job (InferenceRequest + events), optionally POST InferenceResult to Control Plane.
Usage:
  PYTHONPATH=. python scripts/run_layerb_job.py --job jobs/sample_job.json [--post]
  PYTHONPATH=. python scripts/run_layerb_job.py --request-id req-1 --tenant tenant-default --endpoint ep-1 --window-start 1700000000000 --window-end 1700003600000 [--post]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contracts import InferenceRequest
from services.layer_b_inference.pipeline import events_from_episode_events, run_inference
from connectors.control_plane_client import post_result, save_outbox


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=str, help="Path to job JSON (request + events)")
    parser.add_argument("--request-id", type=str, help="Override request_id (with --tenant --endpoint --window-*)")
    parser.add_argument("--tenant", default="tenant-default")
    parser.add_argument("--endpoint", default="ep-001")
    parser.add_argument("--window-start", type=int, default=1700000000000)
    parser.add_argument("--window-end", type=int, default=1700003600000)
    parser.add_argument("--post", action="store_true", help="POST result to Control Plane")
    args = parser.parse_args()

    if args.job:
        path = REPO_ROOT / args.job
        data = json.loads(path.read_text(encoding="utf-8"))
        request = InferenceRequest.model_validate(data.get("request", data))
        events_raw = data.get("events", [])
    else:
        request = InferenceRequest(
            job_id=args.request_id or "cli",
            tenant_id=args.tenant,
            endpoint_id=args.endpoint,
            t0_ms=args.window_start,
            t1_ms=args.window_end,
        )
        events_raw = []

    events = events_from_episode_events(events_raw)
    result = run_inference(request, events, fetch_time_ms=0.0)

    # v1 alignment: request_id, produced_at, model_version (Control Plane expects these)
    payload = result.model_dump()
    payload["request_id"] = getattr(request, "request_id", None) or result.job_id or request.endpoint_id
    payload.setdefault("produced_at", None)
    payload.setdefault("model_version", "v1")

    if args.post:
        ok, err = post_result(result)
        if not ok:
            save_outbox(result, err or "unknown")
            print("POST failed:", err, file=sys.stderr)
            return 1
        print("POST ok:", result.hypothesis.hypothesis_id)
    else:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
