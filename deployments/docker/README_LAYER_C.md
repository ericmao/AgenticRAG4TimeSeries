# Layer C Docker（一鍵整合）

Layer C 提供兩種容器模式：

1. **layerc-demo**（一次性）：啟動後跑一輪 retrieve → analyze → writeback，並 POST 至 Control Plane。
2. **layerc-api**（常駐）：HTTP 服務 `POST /analyze`，由 guacamole-ai API 驅動分析。

## 常駐 API（與 guacamole-ai 一鍵跑、由 API 驅動）

在 **guacamole-ai** 的 `docker-compose.integrated.yml` 已加入 **layerc-api**：

- 常駐服務，對外埠 `8002:8000`；Control Plane API 設有 `ANALYSIS_SERVICE_URL=http://layerc-api:8000`。
- 透過 guacamole-ai 呼叫 **POST /api/v1/analysis/run**（body: `episode_path` 或 `episode`），即轉發至 Layer C 執行分析並回寫 writeback。

```bash
# 兩 repo 同層時
cd guacamole-ai
docker compose -f docker-compose.integrated.yml up -d

# 驅動分析（範例）
curl -X POST http://localhost:8081/api/v1/analysis/run \
  -H "Content-Type: application/json" \
  -d '{"episode_path": "tests/demo/episode_insider_highrisk.json"}'
```

- 若 AgenticRAG4TimeSeries 不在 `../AgenticRAG4TimeSeries`，請設定 `AGENTIC_RAG_PATH`。

## 單獨建置與執行（一次性 demo）

```bash
# 在 AgenticRAG4TimeSeries 目錄
docker build -f deployments/docker/Dockerfile.layer_c -t layerc-demo .
docker run --rm -e CONTROL_PLANE_BASE_URL=http://host.docker.internal:8081 -e CONTROL_PLANE_TOKEN=demo layerc-demo
```

## 單獨建置 API 映像

```bash
docker build -f deployments/docker/Dockerfile.layer_c_api -t layerc-api .
docker run --rm -p 8002:8000 -e CONTROL_PLANE_BASE_URL=http://host.docker.internal:8081 -e CONTROL_PLANE_TOKEN=demo layerc-api
```
