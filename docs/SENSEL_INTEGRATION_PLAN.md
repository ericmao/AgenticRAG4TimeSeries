# SenseL A0BC 四層整合工程規劃

本文件為 **Cursor project / Repo mode** 下之可落地工程規劃與目錄結構調整方案，產出可執行工作包、介面合約、最小可跑 PoC 與 Jira-ready 任務清單。

---

## A) 概念對齊（Architecture Alignment）

### 四層責任邊界與 I/O

| 層級 | 責任 | 輸入 | 輸出 | 所屬 Repo |
|------|------|------|------|------------|
| **Layer A** | Data Plane：事件攝取、正規化、retention、replay。擁有 MQTT/Kafka/Logstash/ETL 與 topic 治理。 | syslog / MQTT / HTTP intake、raw events | NormalizedEvent 寫入 `events.norm.*`、append-only DB | **Control Plane** |
| **Layer 0** | Fast path risk scoring：可解釋、低延遲。定期/即時算 risk_timeseries、trust_delta、top_ttp；可覆寫 Layer B 的 hypothesis。 | normalized_events、hypothesis_store（Layer B 回寫） | RiskSummary、dashboard、API | **Control Plane** |
| **Layer B** | Slow path sequential inference：可 GPU。消費事件、建序列、跑模型/啟發式 → Hypothesis。 | InferenceRequest、NormalizedEvent（由 CP API 或未來 Kafka 提供） | InferenceResult → POST /api/v1/inference/results | **AgenticRAG** |
| **Layer C** | Investigation workflow：Episode → EvidenceSet → triage/hunt/response → writeback/eval。Evidence-only + citations + policy guardrails。 | Episode（手動或由 risk 觸發）、KB/OpenCTI | EvidenceSet、AgentOutput、decision bundle → POST /api/v1/cases/writeback | **AgenticRAG** |

### 明確歸屬

- **僅在 Control Plane repo（A + 0）**  
  - MQTT broker、Kafka-compatible、Postgres、Logstash/Vector pipeline 配置、topic 命名與 retention、`normalized_events` / `risk_timeseries` / `pipeline_metrics` / `connector_status` 表、Dashboard、租戶/站點/設備/政策、Layer 0 scoring 邏輯、`/api/v1/risk/*`、`/api/v1/pipeline/health`、`/api/v1/cti/composition`、**接收** `/api/v1/inference/results` 與 `/api/v1/cases/writeback`。

- **僅在 AgenticRAG repo（B + C）**  
  - Layer B worker（pull jobs、拉事件、推論、POST InferenceResult）、Layer C 管線（cert2episodes、retrieve、analyze、validate、writeback、eval、demo_report）、KB、agents（triage/hunt/response）、contracts 消費端（與 Control Plane contracts v1 一致）。

- **MQTT/Kafka/Logstash/ETL**：屬於 **Layer A**；Layer B/C 為 **consumer**，不擁有 broker/retention/replay 治理權。

- **Layer 0**：fast path、可解釋、低延遲；**Layer B**：slow path、可 GPU；**Layer C**：investigation workflow（Episode → EvidenceSet → agents → writeback/eval）。

---

## B) 合約與介面（Contracts & Interfaces）

跨 repo 必須一致。Control Plane 建立 `contracts/`（或 `libs/contracts`），提供 **TS + Python Pydantic v2**，產出 **JSON Schema v1**。兩 repo 皆加入 **契約一致性檢查（CI）**：生成 JSON Schema 並比對或 snapshot。

完整欄位定義見 **[docs/SENSEL_CONTRACTS_V1.md](SENSEL_CONTRACTS_V1.md)**。摘要：

1. **NormalizedEvent v1** — event_id, ts, tenant_id, endpoint_id, entity_type, entity_id, source, event_type, severity, confidence, fields  
2. **RiskSummary v1** — endpoint_id, risk_score, risk_trend[], risk_trend_direction, trust_delta, top_ttp, last_updated  
3. **Hypothesis v1** — entity_id, window{start,end}, ttp_candidates[], likelihood, trust_delta, risk_score, recommendation, produced_at, model_version  
4. **InferenceRequest v1** — request_id, tenant_id, endpoint_id, window{start,end}, priority, events_query?, callback_url?  
5. **InferenceResult v1** — request_id, tenant_id, endpoint_id, hypothesis, produced_at, model_version  
6. **Episode v1** — episode_id, tenant_id, entity_id, window{start,end}, events: NormalizedEvent[], artifacts[], risk_context?  
7. **EvidenceSet v1** — episode_id, evidence_items[{id, type, content, source_ref, citations[]}], policy_guardrails_version  

TS 與 Python 的 **JSON 形狀必須完全一致**；contracts 版本化於 `v1/`，未來可擴 v2。

---

## C) Control Plane Repo：Layer A + Layer 0 落地骨架

### 建議目錄結構（最小侵入新增）

```
control-plane-repo/
├── contracts/                    # 新增：TS + Python 雙語契約 v1
│   ├── v1/
│   │   ├── normalized_event.ts
│   │   ├── risk_summary.ts
│   │   ├── hypothesis.ts
│   │   ├── inference_request.ts
│   │   ├── inference_result.ts
│   │   ├── episode.ts
│   │   ├── evidence_set.ts
│   │   └── python/               # Pydantic v2 鏡像
│   └── scripts/
│       └── export_schemas.ts     # 輸出 JSON Schema → dist/schemas/v1/
├── deployments/
│   └── layerA/                   # 新增
│       ├── docker-compose.yml    # MQTT + Kafka + Postgres + console
│       ├── .env.example
│       └── README.md             # Topic 命名、replay 策略
├── docs/
│   └── sensel_layer_a0.md        # 新增：Layer A/0 說明、API 清單
├── scripts/                      # 新增或擴充
│   ├── mock_events.py            # 發送 mock → normalized_events
│   ├── create_inference_job.py   # 建立 InferenceRequest（API 或 jobs）
│   └── verify_db_state.py       # 檢查 normalized_events / risk_timeseries
├── src/
│   └── (現有 Next.js + API + DB，新增以下)
│       ├── api/v1/risk/          # GET summary, timeseries
│       ├── api/v1/pipeline/      # GET health
│       ├── api/v1/cti/           # GET composition
│       ├── api/v1/inference/     # POST results（接收 Layer B）
│       ├── api/v1/cases/         # POST writeback（接收 Layer C）
│       └── layer0/               # Risk engine MVP、DB 存取
└── prisma/ or migrations/        # normalized_events, risk_timeseries, hypothesis_store, pipeline_metrics, connector_status
```

### Topic 命名（Layer A）

- `events.raw.*` — 原始攝取  
- `events.norm.*` — 正規化後（NormalizedEvent v1）  
- `signals.layer0.*` — Layer 0 產出訊號  
- `results.layerb.*` — Layer B 回寫結果（可選，或僅 DB）  
- `cases.layerc.*` — Layer C case 相關（可選，或僅 API）

### Layer 0 API 端點（最小集合）

- `GET /api/v1/risk/endpoints/summary?tenant_id=&window_hours=&points=`  
- `GET /api/v1/pipeline/health?tenant_id=&window_min=`  
- `GET /api/v1/cti/composition?tenant_id=&window_hours=`  
- `POST /api/v1/inference/results` — Layer B 回寫 Hypothesis  
- `POST /api/v1/cases/writeback` — Layer C 回寫 decision bundle / patch dry-run  

具體 YAML/TS 與 DB schema 見 **docs/control_plane_spec/**（本 repo 內提供規格，供複製至 Control Plane repo）。

---

## D) AgenticRAG Repo：Layer B + Layer C 落地

### 建議目錄結構（在既有基礎上調整）

```
AgenticRAG4TimeSeries/
├── contracts/                    # 與 Control Plane contracts v1 對齊（或 submodule）
│   ├── v1/
│   │   ├── __init__.py
│   │   ├── normalized_event.py
│   │   ├── risk_summary.py
│   │   ├── hypothesis.py
│   │   ├── inference_request.py
│   │   ├── inference_result.py
│   │   ├── episode.py
│   │   └── evidence_set.py
│   └── export_schemas.py         # 產出 JSON Schema，CI 比對
├── services/
│   └── layer_b_inference/         # 既有
│       ├── pipeline/
│       ├── features/
│       ├── models/
│       ├── worker/
│       └── api/
├── connectors/
│   └── control_plane_client/      # 既有：POST results、outbox
├── src/                          # Layer C 既有
│   ├── contracts/                # Episode/EvidenceSet 與 v1 對齊欄位
│   ├── pipeline/
│   ├── agents/
│   └── ...
├── scripts/
│   ├── run_layerb_job.py         # 新增：跑單一 Layer B job（對接 CP 或 jobs/）
│   └── run_layerc_episode_demo.sh # 新增：從 Episode 到 writeback 的 demo
├── jobs/                         # Layer B pull mode 用
├── deployments/
│   └── docker/
│       ├── Dockerfile.layer_b
│       └── docker-compose.layer_b.yml
└── docs/
    ├── SENSEL_INTEGRATION_PLAN.md
    ├── SENSEL_CONTRACTS_V1.md
    └── control_plane_spec/        # Control Plane 端規格與範例檔（可複製）
```

- **Layer B**：Pull mode 從 `jobs/` 或 polling CP queue；消費事件優先用 Control Plane API 拉取 NormalizedEvent；輸出 POST 至 `/api/v1/inference/results`；GPU 可選（MODEL_DEVICE=auto|cpu|cuda）。  
- **Layer C**：維持現有 Episode pipeline（cert2episodes → retrieve → analyze → validate → writeback → eval → demo_report）；Evidence-only + citations + policy guardrails；預設不呼叫 LLM；回寫 `/api/v1/cases/writeback`。

---

## E) 最小可跑 PoC（End-to-End Demo）

### Demo 流程

1. **Control Plane Layer A**：用 `scripts/mock_events.py` 發送 mock events（syslog 或 HTTP intake）→ 寫入 `normalized_events`。  
2. **Layer 0**：定期計算 `risk_timeseries`；dashboard 可看 risk_score / trust_delta / top_ttp（先 heuristic）。  
3. **Layer B**：建立 InferenceRequest（`jobs/` 或 API）→ 執行 `scripts/run_layerb_job.py` → 回寫 hypothesis_store → Layer 0 覆蓋 top_ttp/likelihood。  
4. **Layer C**：當 risk_score >= threshold（例如 70）時觸發，或手動建立 Episode → 跑 `scripts/run_layerc_episode_demo.sh` → EvidenceSet → triage/hunt/response → writeback decision bundle → 回寫 Control Plane `/api/v1/cases/writeback`。

### Demo 指令（從 0 跑到 end-to-end）

- **Control Plane 端**（於 Control Plane repo 執行）：  
  - `python scripts/mock_events.py`  
  - `python scripts/create_inference_job.py`  
  - `python scripts/verify_db_state.py`  
- **AgenticRAG 端**：  
  - `PYTHONPATH=. python scripts/run_layerb_job.py --job jobs/sample_job.json`  
  - `bash scripts/run_layerc_episode_demo.sh tests/demo/episode_insider_highrisk.json`  

（實際路徑與參數以各 repo 內 scripts 為準。）

---

## F) 工程任務清單（Jira-Ready）

### Epic A: Contracts & CI

| ID | 任務 | 目的 | 主要變更 | 驗收條件（AC） | 風險/依賴 |
|----|------|------|----------|----------------|-----------|
| A1 | Control Plane 建立 contracts/v1（TS） | 定義 NormalizedEvent/RiskSummary/Hypothesis/InferenceRequest/InferenceResult/Episode/EvidenceSet v1 | contracts/v1/*.ts, export_schemas.ts | 可產出 dist/schemas/v1/*.json | 無 |
| A2 | Control Plane 建立 contracts v1 Python 鏡像 | 與 TS JSON 形狀一致 | contracts/v1/python/*.py | Pydantic model_dump() 與 TS 產出 JSON 一致 | 依 A1 |
| A3 | AgenticRAG contracts 對齊 v1 | 消費端與 Control Plane 一致 | contracts/v1/, 或 submodule | CI 通過 schema 比對 | 依 A1 |
| A4 | 兩 repo 契約一致性 CI | 自動比對 JSON Schema | .github/workflows/ or CI config | 任一改動 contracts 時 CI 跑 schema 比對或 snapshot | 依 A2、A3 |

### Epic B: Layer A Data Plane

| ID | 任務 | 目的 | 主要變更 | 驗收條件（AC） | 風險/依賴 |
|----|------|------|----------|----------------|-----------|
| B1 | deployments/layerA docker-compose | MQTT + Kafka + Postgres + console 可起 | deployments/layerA/docker-compose.yml, .env.example | `docker compose up` 成功、topic 可建立 | 無 |
| B2 | Logstash/Vector pipeline placeholder | 說明 raw → NormalizedEvent 流程 | docs 或 config placeholder | 文件說明 events.raw.* → events.norm.*、欄位對應 | 依 A1 |
| B3 | normalized_events 表與寫入 | append-only 儲存 NormalizedEvent | migrations, 寫入邏輯 | mock_events.py 可寫入並可查 | 依 B1、A1 |
| B4 | Replay/audit 策略文件 | 定義 retention、replay 原則 | docs/sensel_layer_a0.md | 文件明確寫出 append-only、replay 策略 | 無 |

### Epic C: Layer 0 Risk Engine

| ID | 任務 | 目的 | 主要變更 | 驗收條件（AC） | 風險/依賴 |
|----|------|------|----------|----------------|-----------|
| C1 | risk_timeseries / pipeline_metrics / connector_status 表 | Layer 0 與 pipeline 監控儲存 | migrations | 表存在、可寫入 | 依 B3 |
| C2 | GET /api/v1/risk/endpoints/summary | 查詢 endpoint risk 摘要 | API route, Layer 0 讀取 | 回傳 RiskSummary v1 形狀 | 依 A1、C1 |
| C3 | Layer 0 scoring MVP（規則/權重） | 可解釋、低延遲 risk_score/trust_delta/top_ttp | layer0/scoring | 定時或觸發寫入 risk_timeseries、dashboard 可顯示 | 依 C1 |
| C4 | POST /api/v1/inference/results 接收與 hypothesis_store | Layer B 回寫寫入 DB、Layer 0 可覆蓋 top_ttp/likelihood | API route, hypothesis_store 表/邏輯 | Layer B POST 後可查、Layer 0 使用其結果 | 依 A1、C3 |
| C5 | GET /api/v1/pipeline/health、GET /api/v1/cti/composition | 營運與 CTI 查詢 | API routes | 回傳符合文件之格式 | 依 C1 |

### Epic D: Layer B Inference Worker

| ID | 任務 | 目的 | 主要變更 | 驗收條件（AC） | 風險/依賴 |
|----|------|------|----------|----------------|-----------|
| D1 | 從 Control Plane API 拉取 NormalizedEvent（MVP） | 消費事件來源 | connectors/control_plane_client 或 layer_b 內 client | 給定 window/endpoint 可取得 events 並跑 inference | 依 C2/C4、A1 |
| D2 | run_layerb_job.py 可跑單 job 並 POST 結果 | 單一 job 端到端 | scripts/run_layerb_job.py | 讀 job 檔或 API、跑推論、POST /api/v1/inference/results 成功 | 依 D1、A3 |
| D3 | Dockerfile + compose 支援 GPU（nvidia runtime） | 部署選項 | Dockerfile.layer_b, docker-compose | 可選 MODEL_DEVICE=cuda、nvidia runtime 可跑 | 無 |
| D4 | 啟發式 scorer 產出 Hypothesis v1 形狀 | 無 ML 時仍可跑 | services/layer_b_inference/models/scorer.py | 產出符合 Hypothesis v1、InferenceResult v1 | 依 A3 |

### Epic E: Layer C Investigation Workflow

| ID | 任務 | 目的 | 主要變更 | 驗收條件（AC） | 風險/依賴 |
|----|------|------|----------|----------------|-----------|
| E1 | Episode v1 與 EvidenceSet v1 欄位對齊 | 與 Control Plane 契約一致 | src/contracts/episode.py, evidence.py 或 contracts/v1 | 可序列化為 v1 JSON、CI 通過 | 依 A3 |
| E2 | POST /api/v1/cases/writeback 客戶端 | Layer C 回寫 decision bundle | connectors/control_plane_client 或 src | writeback 後可呼叫 CP API、dry-run 可選 | 依 C5 |
| E3 | run_layerc_episode_demo.sh 端到端 | 單一 Episode 從 retrieve 到 writeback | scripts/run_layerc_episode_demo.sh | 指定 Episode 可跑通並回寫 CP（或本機檔） | 依 E1、E2 |
| E4 | risk_score >= threshold 觸發 Episode 生成（可選） | 自動化觸發 | Control Plane 或 AgenticRAG 腳本 | 文件或腳本說明如何依 risk 觸發建立 Episode 並跑 Layer C | 依 C3、E3 |

---

## 產出清單

- **兩 repo 建議 tree**：見 C)、D) 小節。  
- **合約 schema v1 摘要**：見 **docs/SENSEL_CONTRACTS_V1.md**。  
- **Demo 指令**：見 E) 小節。  
- **Jira 任務清單**：見 F) 表格（Epic A–E，共 18 項）。

Control Plane 端具體檔案內容（docker-compose、scripts 範例、API 規格）見 **docs/control_plane_spec/**。
