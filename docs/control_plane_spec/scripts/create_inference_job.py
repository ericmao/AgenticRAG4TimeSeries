#!/usr/bin/env python3
"""
Create InferenceRequest v1 (Control Plane repo).
Copy to Control Plane repo: scripts/create_inference_job.py
Output: JSON to stdout or POST to Control Plane API / jobs folder for AgenticRAG.
"""
import json
import os
import sys

def make_inference_request(request_id: str, tenant_id: str, endpoint_id: str, t0_ms: int, t1_ms: int):
    return {
        "request_id": request_id,
        "tenant_id": tenant_id,
        "endpoint_id": endpoint_id,
        "window": {"start": t0_ms, "end": t1_ms},
        "priority": "normal",
    }

if __name__ == "__main__":
    request_id = sys.argv[1] if len(sys.argv) > 1 else "req-001"
    tenant_id = os.environ.get("TENANT_ID", "tenant-default")
    endpoint_id = os.environ.get("ENDPOINT_ID", "ep-001")
    t0_ms = int(os.environ.get("T0_MS", "1700000000000"))
    t1_ms = int(os.environ.get("T1_MS", "1700003600000"))
    req = make_inference_request(request_id, tenant_id, endpoint_id, t0_ms, t1_ms)
    print(json.dumps({"request": req, "events": []}, indent=2))
