# Knowledge Base (KB)

This directory holds markdown and text documents used for C1 retrieval. The retriever chunks documents (700 chars, 120 overlap) and scores by keyword match from the episode and optional hypothesis.

## Contents

- `sop_incident_response.md` – Incident response standard operating procedure
- `policy_acceptable_use.md` – Acceptable use and response policy template

## Usage

Set `KB_PATH=kb/` in `.env` (default). Only `.md` and `.txt` files under this path are loaded. No vector DB is required for the MVP; retrieval is keyword-based and deterministic.

## Query terms

Queries are built from:
- Episode artifacts (ip, domain, hash, file, protocol_cmd, topic)
- Episode sequence tags (e.g. login, lateral, exfil)
- Hypothesis text keywords and suspected_tactics / suspected_intrusion_set
