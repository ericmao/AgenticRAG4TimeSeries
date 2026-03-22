-- 泛用分析執行紀錄（CERT / Wazuh 共用）
-- psql "$DATABASE_URL" -f deployments/sql/analysis_runs_schema.sql

CREATE TABLE IF NOT EXISTS analysis_runs (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source TEXT NOT NULL CHECK (source IN ('cert', 'wazuh')),
    dataset_label TEXT,
    episode_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    target_ip TEXT,
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

CREATE INDEX IF NOT EXISTS idx_analysis_runs_episode ON analysis_runs(episode_id);
CREATE INDEX IF NOT EXISTS idx_analysis_runs_source ON analysis_runs(source);
CREATE INDEX IF NOT EXISTS idx_analysis_runs_created ON analysis_runs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analysis_runs_run_id ON analysis_runs(run_id);
