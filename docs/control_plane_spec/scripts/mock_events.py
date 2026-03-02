#!/usr/bin/env python3
"""
Mock events → normalized_events (Control Plane repo).
Copy to Control Plane repo: scripts/mock_events.py
Requires: DB connection (e.g. POSTGRES_URI), NormalizedEvent v1 shape.
"""
import os
import time
import uuid

# Example: insert into normalized_events (event_id, ts, tenant_id, endpoint_id, entity_type, entity_id, source, event_type, severity, confidence, fields)
# Adjust to your DB client (Prisma/raw SQL).

TENANT = os.environ.get("TENANT_ID", "tenant-default")
ENDPOINT = os.environ.get("ENDPOINT_ID", "ep-001")

def emit_mock_event(ts_ms: int, entity_id: str, event_type: str, source: str = "logon", fields: dict = None):
    return {
        "event_id": f"evt-{uuid.uuid4().hex[:12]}",
        "ts": ts_ms,
        "tenant_id": TENANT,
        "endpoint_id": ENDPOINT,
        "entity_type": "user",
        "entity_id": entity_id,
        "source": source,
        "event_type": event_type,
        "severity": 0.5,
        "confidence": 1.0,
        "fields": fields or {},
    }

if __name__ == "__main__":
    now_ms = int(time.time() * 1000)
    events = [
        emit_mock_event(now_ms - 3600000, "user-001", "logon", "logon", {"host": "PC01", "domain": "internal"}),
        emit_mock_event(now_ms - 1800000, "user-001", "logoff", "logon", {"host": "PC01"}),
        emit_mock_event(now_ms - 900000, "user-001", "connect", "device", {"host": "PC02"}),
    ]
    for e in events:
        print(e)
    # TODO: insert into DB (normalized_events) using Control Plane DB client.
