# Layer B + C LangChain 實作規劃

## 一、目標與架構

- **目標**：Layer B 與 Layer C 以統一契約（v1）與 CERT/sample 資料串通；可選 LangChain 為推論/分析核心；以 **CERT insider 或 sample 資料** 完成實作與測試。
- **架構**：
  - **Layer B**：InferenceRequest + NormalizedEvent → 特徵 → 推論（heuristic / 可選 LangChain）→ Hypothesis v1 → InferenceResult。
  - **Layer C**：Episode（+ 可選 Hypothesis）→ KB 檢索 → 分析（規則 / 可選 LangChain）→ AgentOutput → writeback。
- **資料**：CERT 或 sample 資料 → NormalizedEvent / Episode；測試使用 `tests/demo/`、`tests/samples/` 及合成資料。

---

## 二、控制層資料格式（v1 對齊）

| 契約 | 用途 | 關鍵欄位（v1） |
|------|------|----------------|
| NormalizedEvent | Layer B 輸入、Episode.events | ts_ms, entity, action, artifact, source, confidence, domain |
| InferenceRequest | Layer B 輸入 | request_id, tenant_id, endpoint_id, window (t0_ms, t1_ms) |
| Hypothesis v1 | Layer B 輸出、Layer C 可選 | entity_id, window, ttp_candidates, likelihood, risk_score, recommendation, produced_at, model_version |
| InferenceResult | Layer B 回寫 | request_id, tenant_id, endpoint_id, hypothesis, produced_at, model_version |
| Episode v1 | Layer C 輸入 | episode_id, tenant_id, entity_id, window, events, artifacts, risk_context（可選） |
| EvidenceSet | Layer C retrieve 輸出 | episode_id, items (evidence_items), stats |

---

## 三、實作任務（Epic）

### Epic 1：契約 v1
- 1.1 擴充 Hypothesis v1（entity_id, window, ttp_candidates, likelihood, risk_score, recommendation, produced_at, model_version）
- 1.2 InferenceRequest / InferenceResult 支援 request_id、window 與 v1 一致
- 1.3 Episode 支援 tenant_id, entity_id, window, risk_context（可選）；與現有欄位相容

### Epic 2：CERT / sample 與 Layer B
- 2.1 CERT 或 sample 轉 NormalizedEvent（從 cert_to_episodes 或 data 目錄）
- 2.2 Layer B 單 job：讀 job 檔（request + events）→ run_inference → 寫 InferenceResult
- 2.3 測試：sample events → Layer B → InferenceResult 驗證

### Epic 3：Layer B → Layer C 串接
- 3.1 InferenceResult + events → 建 Episode（含 risk_context）
- 3.2 Layer C retrieve 接受 hypothesis_path（來自 Layer B）
- 3.3 E2E：sample → Layer B → Episode + Hypothesis → Layer C retrieve → analyze → writeback

### Epic 4：測試與紀要
- 4.1 使用 sample CERT 資料跑 E2E，驗證整條 pipeline
- 4.2 產出實作紀要與測試結果文件

---

## 四、實作順序

1. Epic 1（契約 v1）
2. Epic 2（CERT/sample → Layer B）
3. Epic 3（B→C 串接 + E2E 腳本）
4. Epic 4（測試 + 紀要）

---

## 五、檔案與變更摘要

| 變更 | 路徑 |
|------|------|
| 契約 v1 | contracts/inference.py（Hypothesis v1）, src/contracts/episode.py（risk_context 等） |
| CERT/sample → events | scripts/cert_to_normalized_events.py 或擴充 cert_to_episodes |
| Layer B job | scripts/run_layerb_job.py（讀 job JSON，跑 inference，寫結果） |
| B→Episode | scripts/build_episode_from_inference.py 或 pipeline 模組 |
| E2E | scripts/run_e2e_sample_cert.sh 或 run_e2e_sample_cert.py |
| 測試 | tests/test_layer_b_cert_e2e.py 等 |
