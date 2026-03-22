-- MVP: Wazuh IP-scoped Episode → Layer C 分析結果存 PostgreSQL
-- 執行一次即可: psql "$DATABASE_URL" -f deployments/sql/wazuh_mvp_schema.sql

CREATE TABLE IF NOT EXISTS wazuh_mvp_analysis_runs (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    episode_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    target_ip TEXT NOT NULL,
    window_start_ms BIGINT NOT NULL,
    window_end_ms BIGINT NOT NULL,
    alert_count INT NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    evidence_json JSONB,
    agent_outputs_json JSONB,
    issues_json JSONB,
    error_message TEXT,
    writeback_json JSONB,
    layerc_summary JSONB
);

CREATE INDEX IF NOT EXISTS idx_wazuh_mvp_runs_episode ON wazuh_mvp_analysis_runs(episode_id);
CREATE INDEX IF NOT EXISTS idx_wazuh_mvp_runs_target_ip ON wazuh_mvp_analysis_runs(target_ip);
CREATE INDEX IF NOT EXISTS idx_wazuh_mvp_runs_created ON wazuh_mvp_analysis_runs(created_at DESC);
