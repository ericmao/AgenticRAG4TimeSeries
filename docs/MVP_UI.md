# MVP Web UI（analysis_runs）

## 資料庫

先建立表：

```bash
psql "$DATABASE_URL" -f deployments/sql/analysis_runs_schema.sql
psql "$DATABASE_URL" -f deployments/sql/wazuh_ingest_state_schema.sql
psql "$DATABASE_URL" -f deployments/sql/kb_documents_schema.sql
```

（若使用 `insert_analysis_run`，Python 會在首次寫入時 `CREATE TABLE IF NOT EXISTS`。）

## 本機 Docker 一鍵部署

```bash
cd AgenticRAG4TimeSeries
docker compose -f docker-compose.local.yml up -d --build
```

- UI：<http://localhost:8765/runs>
- **KB 瀏覽**（`KB_PATH` 下遞迴 `.md` / `.txt`）：<http://localhost:8765/kb>（**以 `group_key` 分組**；`llm_summary`／`llm_file_summary` 僅來自快取檔，**請求時不呼叫 LLM**）· `GET /api/kb/files` · `GET /api/kb/groups` · `GET /api/kb/raw/{相對路徑}`（子目錄請 URL 編碼 `/`，例如 `cert_candidates%2Ffoo.md`）
- Postgres（主機連線測試）：`postgresql://agentic:agentic@127.0.0.1:5435/agentic`
- 停止：`docker compose -f docker-compose.local.yml down`（加 `-v` 會刪除資料卷）

## 啟動 API（本機直跑、不用 Docker）

```bash
cd AgenticRAG4TimeSeries
export DATABASE_URL=postgresql://...
export PYTHONPATH=.
# 可選：export MVP_UI_API_KEY=secret
uvicorn services.mvp_ui_api.app:app --host 0.0.0.0 --port 8765
```

- 瀏覽器：`http://localhost:8765/runs` · KB：`http://localhost:8765/kb`

### KB 列表上的 LLM 摘要欄位（僅快取）

- **`/kb` 與 `GET /api/kb/groups` 不會在請求時呼叫 LLM**，以免拖慢回應；群組標題仍以規則型 **`heuristic_summary`** 為預設顯示。
- 若存在快取檔，會一併帶出：
  - **`llm_summary`**：來自 **`outputs/.kb_group_llm_cache.json`**（key = `group_key`）。
  - **`llm_file_summary`**：來自 **`outputs/.kb_file_llm_cache.json`**（key = `rel_path`；`content_sha256` 須與目前檔案內容一致才顯示）。
- 寫入上述快取請用**離線／批次腳本**（或自行維護 JSON），勿依賴瀏覽頁面即時觸發 Ollama。

### KB 線上編輯（Postgres，可選）

- **`KB_DB_MODE`**：`off`（預設，僅 `KB_PATH` 檔案）· `merge`（同一路徑以 DB 內容覆蓋檔案）· `db_only`（僅列出／讀取 DB）。非 `off` 時需 **`DATABASE_URL`**；首次寫入亦會 `CREATE TABLE IF NOT EXISTS`。
- **讀**：`/kb`、`/api/kb/files`、`retrieve_from_kb` 皆經 [`src/kb/loader.py`](src/kb/loader.py) 與 UI 一致。
- **寫**：`PUT /api/kb/documents/{相對路徑}`，JSON `body`、`expected_version`（新建覆寫用 `0`）、可選 `editor`；若已設 **`MVP_UI_API_KEY`** 則需 Header **`X-API-Key`**。版本衝突回 **409**。
- **歷史**：`GET /api/kb/documents/{path}/versions?limit=50`
- **編輯頁**：`GET /kb/e/{相對路徑}`（`KB_DB_MODE=off` 時為 404）。列表／預覽頁在啟用 DB 模式時顯示「編輯」連結。
- 匯出至 Git 供 commit 可另撰腳本將 DB 內容寫回 `KB_PATH`（非本次 MVP 必備）。

- 篩選 Wazuh：`/runs?source=wazuh`
- 觸發 CERT 管線（需 `MVP_UI_API_KEY` 時帶 header）：

```bash
curl -X POST http://localhost:8765/api/runs/cert \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $MVP_UI_API_KEY" \
  -d '{"episode_path":"tests/demo/episode_insider_highrisk.json"}'
```

可選欄位 `triage_rules`（字串陣列，例如 `["lateral","burst"]`）：多規則時對同一 evidence 各跑一輪 triage → hunt_planner → response_advisor；未指定時使用環境變數 `TRIAGE_RULES`（逗號分隔）或 episode 的 `sequence_tags` 與內建候選交集，否則 `["default"]`。

- `GET /api/runs/{id}?summary=1`：僅回 id、時間、episode_id、status、`layerc_summary`、`by_rule_keys`（減輕 payload）。
- 詳情頁 `/runs/{id}`：分區顯示 C1 連結、`layerc_summary`、C2 依規則摺疊、C3 連結；完整列資料在可收合區塊。
- Layer C 產物：`outputs/agents/<episode_id>_by_rule.json`（巢狀 `by_rule`）；並寫入第一規則之 `*_triage.json` 等三檔以相容舊流程。

## 相關腳本

- CERT：`scripts/mvp_cert_layer_c.py`（`--triage-rules lateral,burst`）
- Wazuh 單次：`scripts/mvp_wazuh_episode_pg.py`（同上）
- Wazuh 輪詢：`scripts/wazuh_ingest_poll.py`（同上）
