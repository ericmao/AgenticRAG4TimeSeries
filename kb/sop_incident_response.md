# Standard Operating Procedure: Incident Response

## Scope

This SOP covers detection, triage, and containment for security incidents including lateral movement, exfil, and insider threats. It applies to all internal hosts and user activity monitored via Wazuh, Osquery, and endpoint logs.

## Detection and Triage

1. **Alert ingestion**: Alerts from Wazuh, Osquery, and SIEM are correlated by entity (user, host) and time window. Unusual logon times and process creation (e.g. cmd.exe, PowerShell) are tagged for review.
2. **Episode creation**: Each investigation is scoped as an episode with episode_id, run_id, time bounds (t0_ms, t1_ms), entities (user-001, host-002), and artifacts (IP, domain, file path, hash). Sequence tags such as login, lateral, exfil are assigned from rule logic.
3. **Evidence retrieval**: The system retrieves relevant SOP snippets and policy text from this KB, plus optional OpenCTI intelligence. Evidence is ranked by relevance and capped (e.g. 50 items) per episode.

## Containment

- **Isolate** affected hosts when lateral movement or exfil is confirmed. Use the network isolation playbook.
- **Block** malicious IPs and domains at the perimeter; update allowlists only after validation.
- **Collect more data** when the hypothesis is uncertain; prefer watchlist over hard block until triage is complete.

## References

- Policy: acceptable use and response (see policy_acceptable_use.md).
- Escalation: hand off to response_advisor agent for action recommendations and scope.
