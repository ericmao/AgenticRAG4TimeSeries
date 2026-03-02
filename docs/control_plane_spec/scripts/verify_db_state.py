#!/usr/bin/env python3
"""
Verify DB state (Control Plane repo): normalized_events, risk_timeseries.
Copy to Control Plane repo: scripts/verify_db_state.py
"""
import os
import sys

def main():
    # TODO: Connect to Postgres, query:
    # SELECT COUNT(*) FROM normalized_events WHERE tenant_id = ? AND ts >= ?
    # SELECT * FROM risk_timeseries WHERE endpoint_id = ? ORDER BY ts DESC LIMIT 5
    print("normalized_events: (implement with your DB client)")
    print("risk_timeseries: (implement with your DB client)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
