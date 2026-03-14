# Policy: Acceptable Use and Response

## Purpose

This document defines acceptable use expectations and the policy template for response actions (block, isolate, watchlist, collect_more_data). It is used during C1 retrieval to supply evidence for the response_advisor agent.

## Acceptable Use

- Internal systems are for business use. Unusual logon times or access from unexpected locations are investigated as potential insider threat or compromise.
- Endpoint activity (process creation, file access) is logged and may be used for anomaly detection and hunt.

## Response Actions (Allowlist)

1. **block**: Block malicious IP, domain, or hash at perimeter or endpoint. Requires high confidence to avoid business impact.
2. **isolate**: Network isolation of host. Use when lateral movement or active exfil is suspected.
3. **watchlist**: Monitor entity or artifact without blocking. Use when hypothesis is not yet confirmed.
4. **collect_more_data**: Request additional logs or artifacts before taking action. Use when evidence is insufficient.

## Scope and Rollback

- Every response action must document scope (which entities/hosts) and duration. Rollback conditions (e.g. "revert if false positive reported") must be stated.
- Expected impact (e.g. "user cannot access email until cleared") must be recorded.

## Integration

- This policy is retrieved as KB snippets when the episode or hypothesis mentions login, lateral, exfil, block, isolate, or response. Queries include artifact types (ip, domain, hash, file) and sequence tags from the episode.
