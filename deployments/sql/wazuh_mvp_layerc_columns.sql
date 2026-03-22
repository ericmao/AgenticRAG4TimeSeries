-- 既有庫升級：Layer C writeback 與管線摘要
-- psql "$DATABASE_URL" -f deployments/sql/wazuh_mvp_layerc_columns.sql

ALTER TABLE wazuh_mvp_analysis_runs
  ADD COLUMN IF NOT EXISTS writeback_json JSONB,
  ADD COLUMN IF NOT EXISTS layerc_summary JSONB;
