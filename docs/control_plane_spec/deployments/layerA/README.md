# Layer A — Data Plane

- **MQTT Broker**：預設為 **EMQX 5**（1883 MQTT、8083 WebSocket、18083 Dashboard/API）。要讓 guacamole-ai 用 `DATAPLANE_EMQX_URL=http://emqx:18083` 連到本 EMQX：先執行一次 `docker network create sensel-dataplane`，再分別啟動 layerA（`--profile full`）與 guacamole-ai。
- **Topic 命名**：`events.raw.*`（原始）、`events.norm.*`（NormalizedEvent v1）、`signals.layer0.*`、`results.layerb.*`、`cases.layerc.*`。
- **Replay/audit**：append-only `normalized_events` 表；retention 與 replay 策略由 Layer A 擁有，Layer B/C 僅消費。
- **Logstash/Vector**：MVP 可先用 HTTP intake 寫入 DB；pipeline 設定為 placeholder，說明如何從 syslog/MQTT 產出 NormalizedEvent v1 欄位。
