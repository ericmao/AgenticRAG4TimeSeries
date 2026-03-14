# SenseL 兩 Repo 目錄樹與 Demo 指令

## Control Plane Repo — 建議目錄樹（最小侵入）

```
control-plane-repo/
├── contracts/
│   ├── v1/
│   │   ├── normalized_event.ts
│   │   ├── risk_summary.ts
│   │   ├── hypothesis.ts
│   │   ├── inference_request.ts
│   │   ├── inference_result.ts
│   │   ├── episode.ts
│   │   ├── evidence_set.ts
│   │   └── python/           # Pydantic v2 鏡像
│   └── scripts/
│       └── export_schemas.ts
├── deployments/
│   └── layerA/
│       ├── docker-compose.yml
│       ├── .env.example
│       └── README.md
├── docs/
│   └── sensel_layer_a0.md
├── scripts/
│   ├── mock_events.py
│   ├── create_inference_job.py
│   └── verify_db_state.py
├── src/
│   ├── api/v1/risk/
│   ├── api/v1/pipeline/
│   ├── api/v1/cti/
│   ├── api/v1/inference/
│   ├── api/v1/cases/
│   └── layer0/
└── prisma/ or migrations/
```

（可從本 repo 的 `docs/control_plane_spec/` 複製 deployments/layerA 與 scripts 範例。）

---

## AgenticRAG Repo — 目錄樹（現有 + SenseL 新增）

```
AgenticRAG4TimeSeries/
├── contracts/                 # 與 Control Plane v1 對齊
│   ├── v1/                    # (可選) 正式 v1 模組
│   ├── inference.py          # 現有 Layer B 契約
│   └── __init__.py
├── services/layer_b_inference/
├── connectors/control_plane_client/
├── src/                       # Layer C
│   ├── contracts/
│   ├── pipeline/
│   ├── agents/
│   └── ...
├── scripts/
│   ├── run_layerb_job.py     # SenseL: 單一 Layer B job
│   ├── run_layerc_episode_demo.sh  # SenseL: Episode → writeback
│   ├── run_layer_b_worker.py  # 既有: 批次 jobs/
│   └── run_highrisk_demo.sh
├── jobs/
├── deployments/docker/
├── docs/
│   ├── SENSEL_INTEGRATION_PLAN.md
│   ├── SENSEL_CONTRACTS_V1.md
│   ├── SENSEL_REPO_TREES_AND_DEMO.md
│   └── control_plane_spec/    # 供複製到 Control Plane
└── kb/
```

---

## Demo 指令（從 0 跑到 End-to-End）

### 1. Control Plane 端（於 Control Plane repo 執行）

```bash
# 發送 mock events → normalized_events
python scripts/mock_events.py

# 建立 InferenceRequest（輸出 JSON 可存成 job 或呼叫 API）
python scripts/create_inference_job.py req-001 > jobs/req-001.json

# 驗證 DB 狀態
python scripts/verify_db_state.py
```

### 2. AgenticRAG — Layer B

```bash
cd AgenticRAG4TimeSeries
PYTHONPATH=. python3 scripts/run_layerb_job.py --job jobs/sample_job.json
# 若有設定 CONTROL_PLANE_BASE_URL + CONTROL_PLANE_TOKEN：
PYTHONPATH=. python3 scripts/run_layerb_job.py --job jobs/sample_job.json --post
```

### 3. AgenticRAG — Layer C

```bash
cd AgenticRAG4TimeSeries
bash scripts/run_layerc_episode_demo.sh tests/demo/episode_insider_highrisk.json
# 或
bash scripts/run_highrisk_demo.sh   # 含 gen episode + retrieve → writeback → eval → demo_report
```

### 4. 串起 End-to-End（概念）

1. Control Plane: `mock_events.py` → DB 有 normalized_events  
2. Control Plane: Layer 0 定時跑 → risk_timeseries 有值、dashboard 可看  
3. Control Plane: `create_inference_job.py` 產出 job → 放到 AgenticRAG `jobs/`  
4. AgenticRAG: `run_layerb_job.py --post` → POST /api/v1/inference/results → Control Plane 寫入 hypothesis_store、Layer 0 覆蓋 top_ttp  
5. 當 risk_score >= 70（或手動）：建立 Episode → AgenticRAG `run_layerc_episode_demo.sh` → 產出 writeback → POST /api/v1/cases/writeback（需實作 client）

---

## Jira 任務清單（Epic 對應）

| Epic | 任務 ID | 任務名稱 |
|------|---------|----------|
| A | A1 | Control Plane contracts v1 (TS) + JSON Schema 產出 |
| A | A2 | Control Plane contracts v1 Python 鏡像 |
| A | A3 | AgenticRAG contracts 對齊 v1 |
| A | A4 | 兩 repo 契約一致性 CI |
| B | B1 | deployments/layerA docker-compose |
| B | B2 | Logstash/Vector pipeline placeholder |
| B | B3 | normalized_events 表與寫入 |
| B | B4 | Replay/audit 策略文件 |
| C | C1 | risk_timeseries / pipeline_metrics / connector_status 表 |
| C | C2 | GET /api/v1/risk/endpoints/summary |
| C | C3 | Layer 0 scoring MVP |
| C | C4 | POST /api/v1/inference/results 接收與 hypothesis_store |
| C | C5 | GET /api/v1/pipeline/health、GET /api/v1/cti/composition |
| D | D1 | Layer B 從 Control Plane API 拉取 NormalizedEvent |
| D | D2 | run_layerb_job.py 單 job + POST |
| D | D3 | Dockerfile + compose GPU |
| D | D4 | 啟發式 scorer 產出 Hypothesis v1 |
| E | E1 | Episode / EvidenceSet v1 欄位對齊 |
| E | E2 | POST /api/v1/cases/writeback 客戶端 |
| E | E3 | run_layerc_episode_demo.sh 端到端 |
| E | E4 | risk_score >= threshold 觸發 Episode（可選） |

詳細 AC、主要變更、風險見 **docs/SENSEL_INTEGRATION_PLAN.md** 第 F 節。
