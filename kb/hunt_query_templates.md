# Hunt Query Templates

## Purpose

This document provides template queries for hunting inside episode time windows. Use these to expand evidence and pivot from initial entities (users and hosts) to related activity. All templates are plain text; no external systems required.

## Multiple Host Logon

- **Query**: List distinct hosts where the user has logon events in the episode window. Order by first logon time per host.
- **Use case**: Detect lateral movement when one user logs on to many hosts. Compare host count to baseline.
- **Fields**: user, host (computer), activity=logon, ts_ms. Group by user, host; filter by episode t0_ms and t1_ms.

## Logon Burst

- **Query**: Count logon and logoff events per user per time bucket (e.g. hour or day) within the episode.
- **Use case**: Identify burst activity (many logons/logoffs in a short period). Tag episode with "burst" when count exceeds threshold (e.g. 50 events in window).
- **Fields**: user, activity (logon, logoff), ts_ms. Aggregate count by user and bucket.

## Device Churn

- **Query**: List device connect and disconnect events per user and host (or device id) in the episode.
- **Use case**: High device churn may indicate data copy or unstable access. Correlate with logon to see which hosts had both logon and device activity.
- **Fields**: user, computer, activity (connect, disconnect), source=device, ts_ms.

## Pivot Expansion Ideas

- **Host pivot**: From a given host, list all users who had logon or device activity on that host in the window. Use to find shared or compromised hosts.
- **User pivot**: From a given user, list all hosts (and optionally devices) in order of first access. Use to trace lateral path.
- **Time pivot**: From a given ts_ms, list all events (logon, logoff, device) within a small delta (e.g. 15 minutes) to see burst or sequence.
- **Artifact pivot**: From episode artifacts (host, user), run the same templates with artifact values as filters to collect_more_data before response.

## Integration

- These templates are matched to episodes by query terms such as logon, logoff, device, lateral, burst, and host. Retrieval returns relevant KB chunks for triage and hunt planner agents.
