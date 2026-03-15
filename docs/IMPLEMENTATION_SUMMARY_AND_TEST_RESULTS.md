# Layer B + C 實作紀要與測試結果

## 一、實作紀要

### 1. 計畫文件
- **docs/IMPLEMENTATION_PLAN_LAYER_BC_LANGCHAIN.md**：實作規劃（目標、架構、Epic、任務順序、檔案變更摘要）。

### 2. 契約 v1（Epic 1）

| 變更 | 說明 |
|------|------|
| **contracts/inference.py** | NormalizedEvent 新增 `event_id`（可選）。InferenceRequest 新增 `request_id`、`window`、`priority`，`t0_ms`/`t1_ms` 改為可選；新增 `get_start_end_ms()`。Hypothesis 擴充 v1：`entity_id`、`window`、`ttp_candidates`、`likelihood`、`trust_delta`、`risk_score`、`recommendation`、`produced_at`、`model_version`。InferenceResult 新增 `produced_at`、`model_version`。新增 TTPCandidate（未強制使用）。 |
| **services/layer_b_inference/models/scorer.py** | heuristic_scorer 產出 Hypothesis v1：填入 `entity_id`、`window`、`ttp_candidates`、`likelihood`、`risk_score`、`recommendation`、`produced_at`、`model_version`；支援參數 `window`、`entity_id`。 |
| **services/layer_b_inference/pipeline/run.py** | 呼叫 scorer 時傳入 `window`、`entity_id`；InferenceResult 寫入 `produced_at`、`model_version`、`request_id`。 |
| **src/contracts/episode.py** | Episode 新增可選欄位：`tenant_id`、`entity_id`、`window`、`risk_context`。 |

### 3. CERT / Sample 與 Layer B（Epic 2）

| 變更 | 說明 |
|------|------|
| **jobs/sample_job.json** | 新增：含 request（request_id、tenant_id、endpoint_id、t0_ms、t1_ms）與 10 筆 events（CERT 風格），供 Layer B 直接使用。 |
| **scripts/run_layerb_job.py** | 沿用既有邏輯；契約 v1 後產出含 hypothesis.risk_score、produced_at 等。 |

### 4. Layer B → Layer C 串接（Epic 5）

| 變更 | 說明 |
|------|------|
| **src/pipeline/build_episode_from_inference.py** | 新增：`build_episode_from_inference(result, events, ...)` 從 InferenceResult + events 建 Episode（含 risk_context）與 Layer C 用 Hypothesis；`run_build_episode_from_inference(result_path, job_path=..., ...)` 從檔案讀取 result、從 job 讀取 events，寫出 Episode 與 Hypothesis JSON。 |
| **scripts/run_e2e_sample_cert.py** | 新增 E2E：可選 `--episode` 或 `--cert-data`；產出 Layer B job → 跑 Layer B → 從 result 建 Episode + Hypothesis → Layer C retrieve（帶 hypothesis）→ analyze → writeback（dry_run）。 |

### 5. 測試（Epic 4）

| 變更 | 說明 |
|------|------|
| **tests/test_layer_b_cert_e2e.py** | 新增：`test_layer_b_sample_job_produces_inference_result`（Layer B sample job → InferenceResult 形狀與 risk_score）、`test_build_episode_from_inference_result`（result + events → Episode + risk_context）、`test_e2e_episode_from_demo`（demo episode → inference → build episode）。 |

---

## 二、測試結果

### 1. 單元／整合測試（pytest）

指令：
```bash
cd /path/to/AgenticRAG4TimeSeries
PYTHONPATH=. python3 -m pytest tests/test_layer_b_cert_e2e.py -v
```

結果：
- **test_layer_b_sample_job_produces_inference_result**：PASSED（Layer B 產出 InferenceResult，status=success，hypothesis.risk_score 在 [0,1]）
- **test_build_episode_from_inference_result**：PASSED（Episode 含 risk_context，可 model_validate）
- **test_e2e_episode_from_demo**：PASSED（demo episode → inference → build episode，risk_context 存在）

備註：出現 Pydantic 警告 `Field "model_version" has conflict with protected namespace "model_"`，不影響通過，可選在契約中設定 `model_config['protected_namespaces'] = ()` 消除。

### 2. E2E：使用既有 demo episode

指令：
```bash
PYTHONPATH=. python3 scripts/run_e2e_sample_cert.py --episode tests/demo/episode_insider_highrisk.json
```

結果：
- Layer B job 寫入 `outputs/e2e_sample_cert/layerb_job.json`
- Layer B 推論完成，risk_score=1.0，結果寫入 `outputs/e2e_sample_cert/layerb_result.json`
- Episode 與 Hypothesis 寫入 `outputs/e2e_sample_cert/cert-USER0420-highrisk.json` 與 `*_hypothesis.json`
- Layer C retrieve ok（episode_id=cert-USER0420-highrisk，items=22）
- Layer C analyze ok（triage / hunt_planner / response_advisor 各 5 citations）
- Layer C writeback（dry_run）ok，產出 writeback 與 decision_bundle
- **E2E completed successfully.**

### 3. E2E：使用 cert2episodes（合成 CERT 資料）

指令：
```bash
PYTHONPATH=. python3 scripts/run_e2e_sample_cert.py --cert-data data
```

結果：
- cert2episodes 成功，使用第一個 episode（例：cert-USER0083-w4.json）
- Layer B 推論完成，risk_score≈0.345
- Episode + Hypothesis 寫入 `outputs/e2e_sample_cert/`
- Layer C retrieve / analyze / writeback（dry_run）均成功
- **E2E completed successfully.**

---

## 三、產出檔案一覽

| 路徑 | 說明 |
|------|------|
| docs/IMPLEMENTATION_PLAN_LAYER_BC_LANGCHAIN.md | 實作規劃 |
| docs/IMPLEMENTATION_SUMMARY_AND_TEST_RESULTS.md | 本實作紀要與測試結果 |
| contracts/inference.py | 契約 v1 擴充 |
| services/layer_b_inference/models/scorer.py | Hypothesis v1 產出 |
| services/layer_b_inference/pipeline/run.py | 傳入 window/entity_id，寫入 produced_at 等 |
| src/contracts/episode.py | Episode v1 可選欄位 |
| src/pipeline/build_episode_from_inference.py | InferenceResult → Episode + Hypothesis |
| scripts/run_e2e_sample_cert.py | E2E 腳本 |
| jobs/sample_job.json | Layer B sample job |
| tests/test_layer_b_cert_e2e.py | Layer B + CERT E2E 測試 |

---

## 四、後續建議

- 若 Control Plane 已存在：可將契約與 `POST /api/v1/inference/results` 對齊，並在 E2E 中加上 `--post` 與 writeback 至 CP。
- Layer C 可選 LangChain：在 agents 內加入可選 `llm` 路徑（見先前討論），再於 E2E 或 CI 中啟用。
- 使用真實 CERT 資料：將 logon.csv、device.csv 置於 `data/`，再執行 `run_e2e_sample_cert.py --cert-data data` 驗證端到端。
