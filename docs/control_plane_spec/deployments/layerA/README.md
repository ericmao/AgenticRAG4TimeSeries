# Layer A — Data Plane

- **Topic 命名**：`events.raw.*`（原始）、`events.norm.*`（NormalizedEvent v1）、`signals.layer0.*`、`results.layerb.*`、`cases.layerc.*`。
- **Replay/audit**：append-only `normalized_events` 表；retention 與 replay 策略由 Layer A 擁有，Layer B/C 僅消費。
- **Logstash/Vector**：MVP 可先用 HTTP intake 寫入 DB；pipeline 設定為 placeholder，說明如何從 syslog/MQTT 產出 NormalizedEvent v1 欄位。
