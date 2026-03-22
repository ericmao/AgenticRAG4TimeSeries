-- Wazuh Indexer 輪詢游標（Phase 3）
-- psql "$DATABASE_URL" -f deployments/sql/wazuh_ingest_state_schema.sql

CREATE TABLE IF NOT EXISTS wazuh_ingest_state (
    id SERIAL PRIMARY KEY,
    key_name TEXT NOT NULL UNIQUE,
    last_timestamp_ms BIGINT,
    last_pit_id TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
