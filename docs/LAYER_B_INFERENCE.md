# Layer B: Sequential Inference Service

Layer B consumes inference requests (tenant, endpoint, time range / event refs), builds sequence features, runs a model or heuristic scorer, and POSTs a Hypothesis to the Control Plane.

## Env vars

| Variable | Default | Description |
|----------|---------|-------------|
| `CONTROL_PLANE_BASE_URL` | (empty) | Base URL of Control Plane (e.g. `https://cp.example.com`) |
| `CONTROL_PLANE_TOKEN` | (empty) | Bearer token for POST /api/v1/inference/results |
| `TENANT_ID` | `tenant-default` | Default tenant if not in request |
| `GPU_MODE` | `off` | `on` \| `off` (when `on`, use CUDA if available) |
| `MODEL_DEVICE` | `auto` | `auto` \| `cpu` \| `cuda` for model placement |

## Run locally (pull mode)

1. Add job config(s) under `jobs/` (see `jobs/sample_job.json`):
   - `request`: `InferenceRequest` (job_id, tenant_id, endpoint_id, t0_ms, t1_ms, optional event_refs)
   - `events`: list of event dicts (ts_ms, entity, action, artifact, source, confidence, domain)

2. Run worker (no POST if env not set):
   ```bash
   cd /path/to/AgenticRAG4TimeSeries
   PYTHONPATH=. python scripts/run_layer_b_worker.py --jobs-dir jobs --no-post
   ```

3. With Control Plane (POST results):
   ```bash
   export CONTROL_PLANE_BASE_URL=https://your-cp/api
   export CONTROL_PLANE_TOKEN=your-token
   PYTHONPATH=. python scripts/run_layer_b_worker.py --jobs-dir jobs
   ```

Failed POSTs are appended to `outputs/layer_b_outbox/failed_results.jsonl` for later retry.

## Docker (CPU / GPU)

- Build from repo root:
  ```bash
  docker build -f deployments/docker/Dockerfile.layer_b -t layer_b .
  ```
- Run worker with jobs volume:
  ```bash
  docker run --rm -v $(pwd)/jobs:/app/jobs -v $(pwd)/outputs:/app/outputs \
    -e CONTROL_PLANE_BASE_URL -e CONTROL_PLANE_TOKEN layer_b
  ```
- GPU: use `nvidia-docker` or `docker run --gpus all ...` and set `MODEL_DEVICE=cuda` (and install torch in image with `INSTALL_TORCH=1`).

Compose (from repo root):
```bash
docker compose -f deployments/docker/docker-compose.layer_b.yml run --rm layer_b_worker
```

## Health

- Optional API: `GET /healthz` (see `services/layer_b_inference/api/app.py`). Run with uvicorn if needed.
- Worker has no HTTP server by default; use process health checks.

## Contracts (MVP)

- `contracts/` at repo root: `InferenceRequest`, `NormalizedEvent`, `Hypothesis`, `InferenceResult`.
- TODO: Replace with Control Plane package or git submodule when available.
