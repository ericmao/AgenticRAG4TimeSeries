# Layer B Phase 0: Discovery

## Current repo (pre–Layer B)

- **Runtime**: Python 3.11+ (src/, scripts/, tests/).
- **Layer C (existing)**:
  - **Contracts**: `src/contracts/` — Episode, Hypothesis, EvidenceItem, EvidenceSet, AgentOutput, WritebackPatch.
  - **Agents/skills**: `src/agents/` — triage, hunt_planner, response_advisor (rule-based, no LLM in pipeline).
  - **Event ingestion**: No generic event ingestion service. CERT → Episode via `src/pipeline/cert_to_episodes.py` (logon/device CSV → Episode JSON). Episode has `events: list[dict]` (ts_ms, entity, action, artifact, source, confidence, domain).
- **Config**: `src/config.py` + `.env` (OPENAI_API_KEY, OPENCTI_*, KB_PATH, RUN_MODE).

## Planned file/dir changes

| Change | Path |
|--------|------|
| Add | `contracts/` (repo root) — MVP schemas for Layer B (InferenceRequest, NormalizedEvent, Hypothesis, InferenceResult). TODO: replace with package/submodule. |
| Add | `services/layer_b_inference/` — pipeline, features, models, worker, api. |
| Add | `connectors/control_plane_client/` — POST inference results, optional job poll. |
| Add | `deployments/docker/` — Dockerfile(s), docker-compose.yml (CPU + GPU). |
| Add | `docs/LAYER_B_INFERENCE.md` — env vars, run instructions. |
| Add | Tests for schema validation and one inference pipeline test. |
| No change | `src/` Layer C code; coexist with Layer B. |

## Exact new files (MVP)

- `contracts/__init__.py`, `contracts/inference.py` (InferenceRequest, NormalizedEvent, InferenceResult, Hypothesis)
- `contracts/schemas/*.json` (optional JSON Schema export)
- `services/layer_b_inference/__init__.py`
- `services/layer_b_inference/pipeline/__init__.py`, `run.py`
- `services/layer_b_inference/features/__init__.py`, `sequence.py`
- `services/layer_b_inference/models/__init__.py`, `scorer.py`
- `services/layer_b_inference/worker/__init__.py`, `runner.py`
- `services/layer_b_inference/api/__init__.py`, `app.py` (optional FastAPI)
- `connectors/control_plane_client/__init__.py`, `client.py`
- `deployments/docker/Dockerfile.layer_b`, `docker-compose.layer_b.yml`
- `docs/LAYER_B_INFERENCE.md`
- `tests/test_layer_b_contracts.py`, `tests/test_layer_b_pipeline.py`
