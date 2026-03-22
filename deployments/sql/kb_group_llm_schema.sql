-- KB 模板群組 LLM 欄位（意義／使用方向／威脅對應），供 /kb 與批次腳本讀寫
-- 新叢集由 docker-entrypoint-initdb.d 載入；既有 DB 由應用程式 ensure_kb_group_llm_schema 建表

CREATE TABLE IF NOT EXISTS kb_group_llm (
    group_key TEXT PRIMARY KEY,
    meaning TEXT NOT NULL DEFAULT '',
    usage_direction TEXT NOT NULL DEFAULT '',
    threats TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    content_fingerprint TEXT,
    model TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_kb_group_llm_updated ON kb_group_llm (updated_at DESC);
