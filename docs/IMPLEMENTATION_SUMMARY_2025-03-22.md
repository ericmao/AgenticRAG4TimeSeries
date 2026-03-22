# 實作摘要（2025-03-22）

本文件整理當日於 **AgenticRAG4TimeSeries** 相關之 UI、KB、Wazuh、Docker 與 OpenClaw 風格 KB 監看等變更，便於對照與後續維護。

---

## 1. Investigation Graph（調查圖）— MVP Investigation SPA / layerc-graph-ui

### 1.1 版面：全寬 + 外框

- **非 `embedded` 模式**：外層溝槽（`bg-muted` 系）+ 內層單一圓角卡片外框（`rounded-2xl border bg-card shadow-sm ring`），包住 toolbar、摘要卡、篩選與三欄主區。
- **三欄主區**：移除 `main` 上多餘的巢狀框線，改以淺底區分；避免「框中再套框」。
- **`embedded=1`**（Runs 列表內 iframe 三圖）：維持原樣，未改。

**相關檔案**：`packages/layerc-graph-ui/src/components/InvestigationGraphPage.tsx`

### 1.2 案件頁載入／錯誤狀態

- 與列表頁一致：外層溝槽 + 內層圓角白卡，視覺統一。

**相關檔案**：`packages/mvp-investigation-spa/src/main.tsx`

### 1.3 圖表與 Case Navigator 順序

- **大螢幕**：由左至右改為 **圖表 → Episode 列表（Case Navigator）→ Inspector**。
- **小螢幕**：區塊順序同上（圖表在上）。
- 邊框：`lg:border-r` 依新順序調整。

**相關檔案**：`packages/layerc-graph-ui/src/components/InvestigationGraphPage.tsx`

### 1.4 摘要卡不換行

- 標籤（含 `Top confidence`）與數值使用 `whitespace-nowrap`；Triage 的 value / bucket 同一列。
- 整排改 **單列橫向**（`flex flex-nowrap`），窄螢幕可橫向捲動；`md` 以上六格均分。
- **Triage**：若 `triage_level` 與 bucket 字串相同（例如兩段皆 `suspicious`），不重複顯示第二段。

**相關檔案**：`packages/layerc-graph-ui/src/components/InvestigationSummaryCards.tsx`

### 1.5 調查列表：真實資料 + 返回 Runs

- **問題**：`InvestigationListPage` 曾寫死 `MOCK_EPISODE_INDEX`，且 `main.tsx` 未傳入 `fetch('/api/investigations/cases')` 的結果。
- **修正**：列表改為使用 **`episodes={list}`**；後端 `run_row_to_episode_list_entry` 增加 **`source`**（cert / wazuh）；`EpisodeListPanel` 顯示來源並支援搜尋。
- 頁面說明與 **連結至 `/runs`**（完整 Analysis runs）。

**相關檔案**：
- `packages/layerc-graph-ui/src/components/InvestigationListPage.tsx`
- `packages/mvp-investigation-spa/src/main.tsx`
- `services/mvp_ui_api/investigation.py`
- `packages/layerc-graph-ui/src/components/lists/EpisodeListPanel.tsx`

### 1.6 底色灰階

- **`packages/mvp-investigation-spa/src/index.css`**：`@theme` 中 background / card / muted / border 等改為 **oklch 色度 0** 的中性灰；`primary` 仍略帶藍以便連結辨識；`.embed-graph-root` 同步。
- **圖頁**：外層／主區微調 `bg-muted` 比例與 `ring-neutral-950/6`。

**相關檔案**：`packages/mvp-investigation-spa/src/index.css`、`InvestigationGraphPage.tsx`

> **建置**：變更前端後需於 `packages/layerc-graph-ui` 與 `packages/mvp-investigation-spa` 執行 `npm run build`，靜態檔輸出至 `services/mvp_ui_api/static/investigation/`。

---

## 2. KB 列表（`/kb`）— LLM 說明與截斷

- 每檔顯示 **「LLM 說明」** 區塊，內容為 **`llm_file_description`**（由 `llm_file_summary` 截斷而來）。
- **`truncate_llm_file_summary_for_display()`**：空白分詞至多 **200 詞**，全文再 **1200 字元** 上限（防無空白長句）。
- **`GET /api/kb/groups`** 與列表頁共用邏輯，檔案物件含 `llm_file_description`。

**相關檔案**：
- `services/mvp_ui_api/kb_file_llm.py`
- `services/mvp_ui_api/app.py`（`_prepare_kb_groups`、`api_kb_groups`）
- `services/mvp_ui_api/templates/kb_list.html`
- `tests/test_kb_browser.py`（含截斷測試）

---

## 3. KB + Ollama 批次與「持續監看」設計

### 3.1 共用核心：`run_kb_file_llm_refresh`

- **`src/kb/file_llm_refresh.py`**：掃描 KB，對新檔／變更／缺 summary 呼叫 LLM，寫入 `outputs/.kb_file_llm_cache.json`。
- **`scripts/kb_refresh_file_llm.py`**：改為呼叫上述函式，維持 `ok … len=` 等 CLI 輸出。

### 3.2 OpenClaw 風格 agent：`kb_describer`

- **`src/layer_c/agents/kb_describer_agent.py`**：`kb_describer_runner(payload)`、`register_kb_describer(adapter)`，`agent_id`：**`kb_describer`**。
- **`services/mvp_ui_api/agent_dashboard.py`**：`AGENT_CATALOG` 已登錄 **KB Describer**。
- **`src/layer_c/agents/__init__.py`**：匯出 `kb_describer_runner`、`register_kb_describer`。

### 3.3 輪詢腳本：`kb_watch_llm.py`

- 依 **`--interval-sec`**（預設 300）重複執行刷新；**`--once`** 單次；可 **`--max-files-per-cycle`** 限制每輪處理量。
- 預設寫入 **`outputs/kb_watch_activity.json`**（最後一輪統計）；可用 **`--no-activity-file`** 關閉。

**相關檔案**：`scripts/kb_watch_llm.py`

### 3.4 本機執行批次（參考）

- 全量：`PYTHONPATH=. python scripts/kb_refresh_all_llm.py`（內含 file + group LLM）。
- 注意 **`.env` 的 `LLM_MODEL`** 須與本機 `ollama list` 一致（例如已安裝 `llama3:latest` 則勿指向不存在的 `llama3.1:latest`）。

---

## 4. Wazuh 手動分析（文件與設定要點）

- 管線連 **Wazuh Indexer（OpenSearch）REST**，**`WAZUH_INDEXER_URL`** 通常為 **`https://<主機>:9200`**，與 Dashboard 埠不同。
- **`WAZUH_INDEXER_USERNAME` / `WAZUH_INDEXER_PASSWORD`**、`WAZUH_VERIFY_SSL`；另需 **`DATABASE_URL`** 寫入 `analysis_runs`（除非略過寫 DB）。
- UI：**`/runs`** 區塊「手動啟動 Wazuh 分析」→ 背景執行 `mvp_wazuh_episode_pg.py`；若設 **`MVP_UI_API_KEY`** 需帶 **API Key**。

（詳見 repo 內 `env.example`、`docs/MVP_UI.md`。）

---

## 5. Docker 本機部署

- 一鍵重建並啟動：
  ```bash
  docker compose -f docker-compose.local.yml up -d --build
  ```
- **MVP UI**：`http://localhost:8765`；程式碼若以 volume 掛載 repo，Python 變更可即時反映；**Investigation SPA 靜態檔**仍須前述 **npm build** 才會更新。

---

## 6. 測試與指令速查

| 項目 | 指令或位置 |
|------|------------|
| KB 瀏覽器測試 | `pytest tests/test_kb_browser.py` |
| Investigation 前端 build | `packages/layerc-graph-ui` → `packages/mvp-investigation-spa` 各 `npm run build` |
| KB 持續監看 | `PYTHONPATH=. python scripts/kb_watch_llm.py --interval-sec 300` |

---

*以上依當日對話與程式庫狀態整理；若與分支上後續 commit 有差異，以版本庫為準。*
