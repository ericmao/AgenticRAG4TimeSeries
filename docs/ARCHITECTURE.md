# Layer C 架構說明

本文件描述專案中 **Layer C** 管線的整體架構、資料流、模組職責與產物路徑。對應此次更新：CERT → Episode、KB 檢索、C2 Agents、Writeback、Eval、高風險 Demo 與 Docker。

---

## 1. 架構總覽

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Layer C 管線                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  輸入資料            Episode 來源              檢索與分析                    產物   │
│                                                                              │
│  logon.csv     ──┐                                                            │
│  device.csv    ──┼──► cert_to_episodes ──► Episode JSON ──┐                   │
│  (或 synthetic)   │   / gen_insider_highrisk_episode      │                   │
│                   └──────────────────────────────────────┼──► retrieve ──► evidence │
│                                                          │         │        │
│                                                          │         ▼        │
│                                                          │   KB + OpenCTI   │
│                                                          │   (query_strings) │
│                                                          │         │        │
│                                                          │         ▼        │
│                                                          └──► analyze ──► agents │
│                                                                  │        (triage, hunt, response) │
│                                                                  ▼             │
│                                                          validate_and_repair   │
│                                                                  │             │
│                                                                  ▼             │
│                                                          writeback ──► writeback JSON │
│                                                          eval ──► metrics.csv, summary, report │
│                                                          demo_report ──► demo_report.md │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 合約與資料結構

### 2.1 Episode（`src/contracts/episode.py`）

- **episode_id**: 唯一識別，例如 `cert-USER0420-highrisk`、`cert-USER0001-w0`
- **run_id**: 執行批次 ID，例如 `cert-run-1`、`cert-run-hr-1`
- **t0_ms / t1_ms**: 時間視窗起訖（epoch 毫秒）
- **entities**: 該視窗內出現的 user + host（去重）
- **artifacts**: `[{type, value}]`，例如 `{type:"user", value: "USER0420"}`、`{type:"host", value:"PC010"}`
- **sequence_tags**: 由規則產生的標籤，例如 `logon`、`device`、`lateral`、`burst`
- **events**: 事件列表，每筆為 `{ts_ms, entity, action, artifact, source, confidence, domain}`

### 2.2 EvidenceSet（`src/contracts/evidence.py`）

- **episode_id / run_id**: 與 Episode 對應
- **items**: `EvidenceItem[]`，每筆含 `evidence_id`、`source`（kb/opencti）、`kind`、`title`、`body`、`score`、`provenance`
- **stats**: 檢索統計（total_candidates、deduped_count、by_source、by_kind）

### 2.3 AgentOutput（`src/contracts/agent_output.py`）

- **agent_id**: triage | hunt_planner | response_advisor
- **citations**: 引用的 `evidence_id` 列表（至少 3 筆）
- **structured**: 各 agent 的結構化輸出（triage_level、why_now、key_risks、queries、response_plan 等）

---

## 3. Episode 來源（輸入層）

### 3.1 CERT → Episode（`src/pipeline/cert_to_episodes.py`）

| 項目 | 說明 |
|------|------|
| **輸入** | `data_dir` 下 `logon.csv`、`device.csv`（欄位：date, user, computer, activity）；若缺檔則產生最小 synthetic（USER0001/0002、PC001/002） |
| **時間** | `parse_ts_ms(x)` 支援字串 datetime（UTC）、數字（epoch ms/秒） |
| **視窗** | 以每 user 的 `user_min_ts` 為基準，`window_index = floor((ts_ms - user_min_ts) / window_ms)`，確定性分窗 |
| **輸出** | 每個 (user, window_index) 一筆 Episode，寫入 `out_dir/{episode_id}.json`，並以 `Episode.model_validate` 驗證 |

**CLI**：`python -m src.cli cert2episodes --data_dir ./data --out_dir outputs/episodes/cert --window_days 7 --run_id cert-run-1`

### 3.2 高風險情境產生器（`scripts/gen_insider_highrisk_episode.py`）

| 項目 | 說明 |
|------|------|
| **輸入** | `--out_path`、`--user`、`--hosts`（逗號分隔）、`--t0_ms`、`--window_ms`、`--burst_events`、`--device_churn` |
| **輸出** | 單一 Episode：4 筆 logon（lateral）、burst_events 筆交替 logon/logoff、device_churn 對 connect/disconnect；`episode_id=cert-<USER>-highrisk` |

---

## 4. 檢索層（C1）

### 4.1 流程（`src/pipeline/retrieve_evidence.py`）

1. 載入 Episode（及可選 Hypothesis）
2. **build_queries**（`src/retrievers/query_builder.py`）：由 `episode.artifacts`、`episode.sequence_tags`、hypothesis 關鍵字產生 `query_strings`（確定性排序）
3. **retrieve_from_kb**：從 `KB_PATH`（預設 `kb/`）載入所有 `*.md`、`*.txt`，切 chunk（約 700 字、overlap 120），以關鍵字評分，產出 EvidenceItem 候選
4. **retrieve_from_opencti**：若設定 OPENCTI_URL/TOKEN 且 RUN_MODE≠dry_run 則呼叫；否則回傳 []
5. **assemble_evidence**：合併、去重、依 score 與 source 排序，取前 max_items（預設 50），寫入 `outputs/evidence/<episode_id>.json`

### 4.2 知識庫（KB）

- **路徑**：由 `KB_PATH`（env）指定，預設 `kb/`
- **內容**：純 Markdown/文字，無外部依賴；檢索為關鍵字匹配、無向量庫
- **此次新增**：
  - **sop_insider_anomaly.md**：異常 logon、lateral 訊號、升級條件、triage 步驟
  - **hunt_query_templates.md**：多主機 logon、logon burst、device churn、pivot 範本
  - **response_policy_guardrails.md**：watchlist / isolate / collect_more_data / rollback 條件

---

## 5. 分析層（C2 Agents）

### 5.1 執行流程（`src/pipeline/run_agents.py`）

- 依序執行 **triage**、**hunt_planner**、**response_advisor**
- 輸入：Episode + EvidenceSet；**無網路呼叫**（rule-based / stub，非 LLM）
- 輸出寫入 `outputs/agents/<episode_id>_<agent_id>.json`
- 每個 agent 至少 3 筆 citations（evidence_id）

### 5.2 各 Agent 職責

| Agent | 模組 | 輸出重點 |
|-------|------|----------|
| **triage** | `src/agents/triage.py` | triage_level（critical/suspicious/noise）、why_now、top_evidence、key_risks；依 sequence_tags 與 evidence 關鍵字推導 |
| **hunt_planner** | `src/agents/hunt_planner.py` | 查詢思路、預期發現、next_required_data；引用 KB 中的 hunt 範本 |
| **response_advisor** | `src/agents/response_advisor.py` | 回應計畫（watchlist/isolate/collect_more_data）、guardrails、rollback_conditions；引用 KB 政策 |

### 5.3 驗證與修復（`src/pipeline/validate_and_repair.py`）

- 檢查 citations 是否皆存在於 EvidenceSet
- 若未通過可依 repair_hint 重跑或寫入 `outputs/issues/<episode_id>.json`

---

## 6. Writeback 層（C3）

- **模組**：`src/pipeline/writeback_pipeline.py`、`src/memory/opencti_writeback.py`
- **輸入**：Episode、EvidenceSet、AgentOutputs
- **輸出**：`outputs/writeback/<episode_id>.json`（patch：sightings、relationships、notes）、`outputs/audit/decision_bundle_<episode_id>.json`
- **模式**：dry_run（僅記錄）、review、auto（需 OpenCTI 設定）

---

## 7. 評估與報告

### 7.1 Eval（`src/eval/run_eval.py`）

- 對 `episodes_dir` 下 Episode JSON 依序執行：retrieve → agents → validate_and_repair → writeback
- 產出：`outputs/eval/metrics.csv`（含 episode_id、pass、evidence_items、citations_*、egr_overall、ucr_proxy、各階段 ms）、`summary.json`、`report.md`

### 7.2 Demo Report（`src/pipeline/demo_report.py`）

- 依單一 episode 彙總 evidence、agents、writeback、eval 等產物，產生 `outputs/demo/demo_report.md`

---

## 8. CLI 一覽

| 指令 | 說明 |
|------|------|
| `validate` | 載入 config、寫 audit log、匯出 schema、驗證 sample episode |
| `retrieve --episode <path>` | 建 EvidenceSet，寫入 outputs/evidence/ |
| `analyze --episode <path>` | 若無 evidence 先 retrieve，再跑 triage/hunt/response，寫入 outputs/agents/ |
| `writeback --episode <path> [--mode dry_run\|review\|auto]` | 產出 writeback patch 與 decision bundle |
| `eval --episodes_dir <dir> [--limit N]` | 批次跑 retrieve→analyze→writeback，寫 metrics/summary/report |
| `cert2episodes [--data_dir] [--out_dir] [--window_days] [--run_id]` | CERT → Episode JSON 批次產出 |
| `demo_report --episode <path>` | 產生 outputs/demo/demo_report.md |

---

## 9. 目錄與產物路徑

```
專案根目錄/
├── kb/                          # 知識庫（*.md, *.txt）
│   ├── sop_insider_anomaly.md
│   ├── hunt_query_templates.md
│   ├── response_policy_guardrails.md
│   ├── sop_incident_response.md
│   └── policy_acceptable_use.md
├── data/                        # CERT 原始資料（logon.csv, device.csv）
├── src/
│   ├── contracts/               # Episode, Evidence, AgentOutput, Hypothesis, Writeback
│   ├── pipeline/                # retrieve_evidence, cert_to_episodes, run_agents, validate_and_repair, writeback_pipeline, demo_report
│   ├── retrievers/              # kb, opencti, query_builder, assemble
│   ├── agents/                  # triage, hunt_planner, response_advisor, common
│   ├── eval/                    # run_eval, metrics
│   └── config.py                # .env：OPENAI_API_KEY, OPENCTI_*, KB_PATH, RUN_MODE
├── scripts/
│   ├── gen_insider_highrisk_episode.py
│   └── run_highrisk_demo.sh     # 一鍵：gen → retrieve → analyze → writeback → eval → demo_report
├── tests/demo/                  # 高風險 demo 用 Episode（如 episode_insider_highrisk.json）
├── outputs/
│   ├── evidence/                # <episode_id>.json
│   ├── agents/                  # <episode_id>_triage.json, _hunt_planner.json, _response_advisor.json
│   ├── writeback/               # <episode_id>.json
│   ├── eval/                    # metrics.csv, summary.json, report.md
│   ├── demo/                    # demo_report.md
│   └── audit/                   # *.jsonl, decision_bundle_*.json
├── Dockerfile
├── docker-compose.yml
└── docs/
    └── ARCHITECTURE.md          # 本文件
```

---

## 10. 設定與依賴

- **環境變數**（`.env`，見 `env.example`）：`OPENAI_API_KEY`、`OPENCTI_URL`、`OPENCTI_TOKEN`、`KB_PATH`、`RUN_MODE`、`PROMPT_VERSION`
- **RUN_MODE=dry_run**（預設）：不呼叫 OpenAI、不呼叫 OpenCTI；Layer C 管線中的 agents 為規則/stub，不需 LLM
- **Docker**：單一服務 `app`，掛載 `outputs/`、`kb/`、`data/`，`env_file: .env`

以上為此次更新後之 Layer C 架構與產物路徑說明。
