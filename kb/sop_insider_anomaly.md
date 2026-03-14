# Standard Operating Procedure: Insider Anomaly Triage

## Scope

This SOP defines how to detect, triage, and escalate anomalous logon and lateral movement signals for insider-threat and CERT-style episodes. It applies to activity bounded by episode time windows and entities (users and hosts).

## Anomalous Logon Signals

- **Off-hours logon**: Logon outside business hours for the entity or role. Correlate with shift patterns and policy.
- **Multiple host logon**: Same user logging on to multiple distinct hosts within a short window. Count distinct hosts per user per episode.
- **Logon burst**: Unusually high number of logon or logoff events in a single time window. Use a configurable threshold (e.g. events per hour or per day) to flag.
- **First-time host**: User logging on to a host they have not used before in the observed history. Requires baseline or allowlist.

## Lateral Movement Signals

- **Lateral** activity is indicated when a user accesses two or more distinct hosts in the same episode. Tag episodes with sequence_tag "lateral" when distinct host count is at least two.
- **Device churn**: Repeated connect and disconnect (device activity) across multiple hosts or USB/removable devices. Correlate with logon to identify pivot points.
- **Pivot expansion**: After initial access, expansion to additional hosts or devices within the window. Use hunt queries to trace the path.

## Escalation Criteria

- Escalate to human review when: triage_level is high, lateral movement is present, or logon burst exceeds threshold.
- Escalate when key_risks mention containment, isolate, or exfil and evidence supports it.
- Do not escalate on low-confidence or single-event anomalies without corroborating evidence.

## Triage Steps

1. **Ingest**: Load episode entities (users, hosts) and events (logon, logoff, device). Ensure all events have ts_ms within the episode window.
2. **Tag**: Assign sequence_tags from activity types (logon, logoff, device) and heuristics (lateral, burst). Use deterministic rules.
3. **Retrieve**: Pull KB and optional intel for episode; rank evidence by relevance to entities and tags.
4. **Assess**: Compare event patterns to anomalous logon and lateral movement signals above. Document why_now and key_risks.
5. **Decide**: Set triage_level (low, medium, high) and recommend watchlist, collect_more_data, or escalate to isolate/block per response policy.

## References

- Hunt query templates: hunt_query_templates.md.
- Response guardrails: response_policy_guardrails.md.
