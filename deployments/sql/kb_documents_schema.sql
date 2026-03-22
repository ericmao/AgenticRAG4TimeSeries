-- KB 線上編輯：文件本體 + 版本歷史（append-only）
-- 由應用程式 ensure_kb_documents_schema 或手動執行此檔

CREATE TABLE IF NOT EXISTS kb_documents (
    id SERIAL PRIMARY KEY,
    rel_path TEXT NOT NULL UNIQUE,
    body TEXT NOT NULL,
    version INT NOT NULL DEFAULT 1,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_by TEXT
);

CREATE TABLE IF NOT EXISTS kb_document_versions (
    id SERIAL PRIMARY KEY,
    document_id INT NOT NULL REFERENCES kb_documents(id) ON DELETE CASCADE,
    version INT NOT NULL,
    body TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    editor TEXT,
    note TEXT,
    UNIQUE (document_id, version)
);

CREATE INDEX IF NOT EXISTS idx_kb_document_versions_doc_ver
    ON kb_document_versions(document_id, version DESC);
