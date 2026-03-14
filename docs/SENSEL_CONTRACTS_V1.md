# SenseL Contracts v1 — 跨 Repo 一致 JSON 形狀

TS 與 Python（Pydantic v2）產出的 **JSON 形狀必須完全一致**。以下為 v1 欄位定義與範例，兩 repo 實作時請依此產出 JSON Schema 並做 CI 比對。

---

## 1. NormalizedEvent v1

| 欄位 | 型別 | 必填 | 說明 |
|------|------|------|------|
| event_id | string | Y | 唯一識別 |
| ts | number (ms) | Y | 時間戳 |
| tenant_id | string | Y | 租戶 |
| endpoint_id | string | Y | 端點/站點 |
| entity_type | string | Y | user / host / device / other |
| entity_id | string | Y | 實體 ID |
| source | string | Y | logon / device / wazuh / osquery / ... |
| event_type | string | Y | 事件型別 |
| severity | number | N | 0–1 |
| confidence | number | N | 0–1，預設 1 |
| fields | object | N | 擴充欄位（如 artifact: {host}, action, domain） |

**JSON 範例：**
```json
{
  "event_id": "evt-001",
  "ts": 1700000000000,
  "tenant_id": "tenant-default",
  "endpoint_id": "ep-001",
  "entity_type": "user",
  "entity_id": "user-001",
  "source": "logon",
  "event_type": "logon",
  "severity": 0.5,
  "confidence": 1.0,
  "fields": { "host": "PC01", "action": "logon", "domain": "internal" }
}
```

---

## 2. RiskSummary v1

| 欄位 | 型別 | 必填 | 說明 |
|------|------|------|------|
| endpoint_id | string | Y | 端點 ID |
| risk_score | number | Y | 0–100 或 0–1（需一致） |
| risk_trend | number[] | N | 時間序列點 |
| risk_trend_direction | string | N | up / down / stable |
| trust_delta | number | N | 信任變化量 |
| top_ttp | array | N | [{ technique_id, technique_name, tactic, confidence }] |
| last_updated | string (ISO) | Y | 最後更新時間 |

---

## 3. Hypothesis v1

| 欄位 | 型別 | 必填 | 說明 |
|------|------|------|------|
| entity_id | string | Y | 主要實體 |
| window | { start, end } | Y | ms 或 ISO |
| ttp_candidates | array | Y | [{ technique_id, technique_name, tactic, confidence }] |
| likelihood | number | N | 0–1 |
| trust_delta | number | N | |
| risk_score | number | N | |
| recommendation | string | N | |
| produced_at | string (ISO) | Y | |
| model_version | string | N | |

---

## 4. InferenceRequest v1

| 欄位 | 型別 | 必填 | 說明 |
|------|------|------|------|
| request_id | string | Y | 唯一請求 ID |
| tenant_id | string | Y | |
| endpoint_id | string | Y | |
| window | { start, end } | Y | ms |
| priority | string | N | high / normal / low |
| events_query | object | N | 可選查詢條件 |
| callback_url | string | N | 可選回調 |

---

## 5. InferenceResult v1

| 欄位 | 型別 | 必填 | 說明 |
|------|------|------|------|
| request_id | string | Y | 對應 InferenceRequest |
| tenant_id | string | Y | |
| endpoint_id | string | Y | |
| hypothesis | Hypothesis v1 | Y | |
| produced_at | string (ISO) | Y | |
| model_version | string | N | |

---

## 6. Episode v1（Layer C）

| 欄位 | 型別 | 必填 | 說明 |
|------|------|------|------|
| episode_id | string | Y | |
| tenant_id | string | Y | |
| entity_id | string | Y | 主要實體 |
| window | { start, end } | Y | ms |
| events | NormalizedEvent[] | Y | 或與現有 events[] 相容之形狀 |
| artifacts | array | Y | [{ type, value }] |
| risk_context | object | N | 來自 Layer 0 的 risk 摘要 |

---

## 7. EvidenceSet v1（Layer C）

| 欄位 | 型別 | 必填 | 說明 |
|------|------|------|------|
| episode_id | string | Y | |
| evidence_items | array | Y | [{ id, type, content, source_ref, citations[] }] |
| policy_guardrails_version | string | N | 政策版本 |

---

## 版本與 CI

- 所有 schema 置於 **contracts/v1/**，未來可擴 **v2**。
- Control Plane：`npm run build:schemas` 或 `ts-node scripts/export_schemas.ts` → `dist/schemas/v1/*.json`。
- AgenticRAG：`python -m contracts.export_schemas` → `outputs/schemas/v1/*.json`。
- CI：兩邊產出之 JSON Schema 比對或 snapshot 比對，任一變更即失敗並需審查。
