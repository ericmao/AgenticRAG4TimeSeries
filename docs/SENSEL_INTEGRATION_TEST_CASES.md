# SenseL Phase 1 整合測試案例

本文件提供 **2 個整合測試案例**，用於驗證 Phase 1：Layer C writeback 雙通道與 guacamole-ai Watch Dataplane MQTT。

---

## 前置條件

- **guacamole-ai**（Control Plane）已啟動，且 `POST /api/v1/cases/writeback` 可用。
- **sensel-dataplane**（可選）：EMQX 已啟動且可連線；若未部署，測試案例 1 仍可僅驗證 API 路徑，測試案例 2 會回傳「未設定 dataplane」或空清單。
- **AgenticRAG4TimeSeries**：`connectors/control_plane_client.post_writeback` 可用；環境變數 `CONTROL_PLANE_BASE_URL`、`CONTROL_PLANE_TOKEN` 已設定（測試案例 1）。

---

## 測試案例 1：Layer C writeback 經 API 與（可選）MQTT 結果一致

**目的**：驗證同一筆 WritebackPatch 經 **HTTP API** 送達 guacamole-ai 後，Control Plane 正確接收並回傳 `ok`；若已部署 MQTT bridge，可選驗證經 MQTT 送達的結果與 API 一致。

**步驟**：

1. **1.1 API 路徑**
   - 使用 AgenticRAG 的 `post_writeback` 或直接 `POST` 一筆 WritebackPatch 到 `{CONTROL_PLANE_BASE_URL}/api/v1/cases/writeback`。
   - Body 範例：
     ```json
     {
       "episode_id": "test-ep-001",
       "run_id": "run-001",
       "mode": "dry_run",
       "sightings": [],
       "relationships": [],
       "notes": [{"type": "note", "content": "Phase 1 integration test"}],
       "stats": {}
     }
     ```
   - Header：`Authorization: Bearer {CONTROL_PLANE_TOKEN}`、`Content-Type: application/json`。
   - **驗收**：HTTP 200，response body 含 `"ok": true`、`"episode_id": "test-ep-001"`、`"run_id": "run-001"`。

2. **1.2（可選）MQTT 路徑**
   - 若 sensel-dataplane 已部署 writeback bridge：向 EMQX 主題 `sensel/cases/writeback` 發佈上述同一 JSON。
   - **驗收**：bridge 轉發後，guacamole-ai 日誌或 DB 可見到同一 episode_id/run_id 的 writeback 記錄，且結果與 1.1 一致（例如同為成功）。

3. **1.3 比對**
   - 若 1.1 與 1.2 皆執行：兩次寫入的 episode_id/run_id 對應的處理結果應一致（皆成功或皆為相同錯誤）。

**執行方式**（API 路徑）：

```bash
# 從 AgenticRAG4TimeSeries 目錄
export CONTROL_PLANE_BASE_URL=http://localhost:8000  # 依實際 CP 位址調整
export CONTROL_PLANE_TOKEN=your-token

python -c "
import os, json
from connectors.control_plane_client import post_writeback
patch = {
    'episode_id': 'test-ep-001',
    'run_id': 'run-001',
    'mode': 'dry_run',
    'sightings': [],
    'relationships': [],
    'notes': [{'type': 'note', 'content': 'Phase 1 integration test'}],
    'stats': {}
}
ok, err = post_writeback(patch)
assert ok, err
print('OK: writeback accepted')
"
```

---

## 測試案例 2：guacamole-ai Dataplane MQTT 分頁 — topics API 與 Dashboard 顯示

**目的**：驗證 guacamole-ai **Watch Dataplane MQTT** 功能：後端 `GET /api/v1/dataplane-mqtt/topics` 可回傳 dataplane EMQX 的 topic 清單（或未設定時回傳明確狀態）；Dashboard「Dataplane MQTT」分頁可載入並顯示該清單。

**步驟**：

1. **2.1 後端 API**
   - 對 guacamole-ai 發送：`GET {CONTROL_PLANE_BASE_URL}/api/v1/dataplane-mqtt/topics`。
   - 需帶入與 CTI Dashboard 相同的認證（例如 SMB session cookie 或 Bearer）。
   - **驗收（已設定 DATAPLANE_EMQX_URL）**：HTTP 200，body 含 `topics` 陣列（可為空）；可選含 `broker`、`error` 等欄位供除錯。
   - **驗收（未設定）**：HTTP 200 或 503，body 含明確說明（例如 `configured: false` 或 `error`），前端不崩潰。

2. **2.2 Dashboard 分頁**
   - 登入 CTI 儀表板，於分頁下拉選單選擇「Dataplane MQTT」。
   - **驗收**：分頁切換成功，顯示「Topic 清單」區塊；呼叫 `loadDataplaneMqttTopics()` 後，清單區域顯示 API 回傳的 topics 或「未設定 dataplane / 無資料」等提示。
   - 可選：在「訊息預覽」輸入 topic 篩選（如 `sensel/#`）後，若有 `GET /api/v1/dataplane-mqtt/messages` 或 SSE，能顯示近期訊息或「尚未訂閱」提示。

**執行方式**（僅 API，無 UI）：

```bash
# 依實際 CP 位址與認證方式調整
curl -s -H "Authorization: Bearer YOUR_TOKEN" \
  "http://localhost:8000/api/v1/dataplane-mqtt/topics" | jq .
```

**預期範例（已設定 EMQX）**：

```json
{
  "configured": true,
  "topics": ["sensel/default/policy/blacklist", "sensel/cases/writeback"],
  "broker": "http://emqx:18083"
}
```

**預期範例（未設定）**：

```json
{
  "configured": false,
  "topics": [],
  "error": "DATAPLANE_EMQX_URL not set"
}
```

---

## 總結

| 案例 | 驗證重點 | 必要條件 |
|------|----------|----------|
| **1** | Layer C writeback 經 API（與可選 MQTT）正確送達 guacamole-ai，結果一致 | guacamole-ai 運行；案例 1.2 需 dataplane bridge |
| **2** | Dataplane MQTT 後端 API 與 CTI 分頁可載入 topic 清單或未設定狀態 | guacamole-ai 運行；已實作 G3/G4；EMQX 可選 |

兩項測試通過即可視為 Phase 1 整合（Layer C 雙通道 + Watch Dataplane MQTT）驗收達標。

---

## 含 sensel-dataplane 的本地部署與 MQTT 監看驗證

本節為**實際啟動 dataplane（EMQX）+ 設定 EMQX URL + MQTT 發布與監看**的步驟，用於本地整合測試。

### 1. 終止本機佔用 port 的服務

- 若曾啟動 EdgeX 或其它 MQTT broker，先停止以釋放 **1883**、**18083**：
  ```bash
  docker stop edgex-mqtt-broker 2>/dev/null
  ```
- 若有舊的 layerA 容器：`cd sensel-dataplane/deployments/layerA && docker compose down`

### 2. 本地部署 dataplane（僅 EMQX）

- 使用 sensel-dataplane 的 layerA 只啟動 EMQX（避免與本機 5432、8080 衝突）：
  ```bash
  cd sensel-dataplane/deployments/layerA
  docker compose up -d emqx
  ```
- 確認健康：`curl -s -o /dev/null -w "%{http_code}" http://localhost:18083/api/v5/status` 應為 `200`。

### 3. 設定 guacamole-ai 的 EMQX URL 與 API 認證

- 在 guacamole-ai 的 `sensel_control_plane/.env` 設定：
  ```bash
  DATAPLANE_EMQX_URL=http://localhost:18083
  ```
- **EMQX 5 REST API 使用 API Key 認證**（非 Dashboard 登入帳密）。請在 EMQX Dashboard 建立 API Key：
  1. 開啟 http://localhost:18083 ，登入（預設 admin / public）。
  2. 左側 **System** → **API Keys**，新增一組 Key（例如 name: `control-plane`），記下 **Key** 與 **Secret**。
  3. 在 `.env` 設定：
     ```bash
     DATAPLANE_EMQX_USERNAME=<上方的 Key>
     DATAPLANE_EMQX_PASSWORD=<上方的 Secret>
     ```
- 重啟 guacamole-ai 後，`GET /api/v1/dataplane-mqtt/topics` 才會回傳 topic 清單。

### 4. MQTT 發布測試（可選）

- 向 dataplane EMQX 發佈一筆測試訊息（預設允許匿名 publish）：
  ```bash
  mosquitto_pub -h localhost -p 1883 -t "sensel/test/monitoring" -m '{"test":"phase1"}'
  ```
- 若已設定 API Key，在 Dashboard 的 **Monitoring** → **Topics** 或透過 `GET /api/v1/dataplane-mqtt/topics` 應可看到 `sensel/test/monitoring`（或其它已發佈的 topic）。

### 5. 驗收

- 呼叫 guacamole-ai：`curl -s -H "Authorization: Bearer <token>" "http://localhost:8081/api/v1/dataplane-mqtt/topics" | jq .`
- **驗收**：`configured: true`，且 `topics` 陣列中出現預期 topic（如 `sensel/test/monitoring`）。
- 或登入 CTI 儀表板 → 分頁「Dataplane MQTT」→ 重新整理，應顯示 topic 清單。
