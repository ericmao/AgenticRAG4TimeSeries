# Response Policy Guardrails

## Purpose

This document defines when to apply each response action (watchlist, isolate, collect_more_data) and rollback conditions. Use it as evidence for the response_advisor agent so that recommendations stay within policy.

## When to Watchlist

- **watchlist** is used when the hypothesis is not yet confirmed or confidence is low. Add the entity (user or host) to a watchlist for enhanced monitoring without blocking access.
- Use watchlist when lateral movement or logon burst is suspected but evidence is insufficient for isolate or block.
- Use watchlist when triage_level is medium and key_risks do not yet justify containment.

## When to Isolate

- **isolate** is used when lateral movement or active exfil is confirmed with sufficient evidence. Isolate affected hosts from the network to prevent further spread.
- Do not isolate on a single anomaly. Require multiple corroborating evidence items (e.g. logon to multiple hosts plus device churn or burst).
- Document which hosts are in scope and the expected duration. State rollback conditions in the same decision.

## When to Collect More Data

- **collect_more_data** is used when evidence is insufficient to choose between watchlist and isolate. Request additional logs, artifacts, or hunt results before taking action.
- Prefer collect_more_data over block or isolate when triage is incomplete or when the episode has few evidence items.
- After collecting more data, re-run triage and response with the updated evidence set.

## Rollback Conditions

- Every response action that affects availability (isolate, block) must define rollback conditions. Standard rollback conditions include:
  - Revert if false positive is reported by the asset owner or investigation lead.
  - Revert if evidence does not support the action after review.
  - Revert after a stated time window if no further findings.
- Document rollback in the response plan so that operators can restore service safely.

## Summary Table

| Action             | When to use                          | Rollback condition                    |
|--------------------|--------------------------------------|--------------------------------------|
| watchlist          | Unconfirmed hypothesis, medium risk | Remove from watchlist when cleared   |
| isolate            | Confirmed lateral or exfil           | Revert if false positive or no support |
| collect_more_data  | Insufficient evidence                | N/A (no impact yet)                   |

## Integration

- This policy is retrieved when episode or hypothesis mentions watchlist, isolate, collect_more_data, or rollback. Citations to this document support guardrails and scope in the response plan.
