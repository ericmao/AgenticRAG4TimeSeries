"""
KB 目錄瀏覽：解析 KB_PATH（相對 REPO_ROOT）、列舉 .md/.txt、安全解析檔案路徑。
模板分組：正規化 episode／Hosts／Event count 等可變欄位後取 SHA-256 前 12 字元為 group_key。
"""
from __future__ import annotations

import hashlib
import html
import re
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException

# 讀取全文做分組時的上限（避免異常大檔）
_MAX_BYTES_GROUPING = 2_000_000

# CERT 候選等檔案中的 **Sequence tags** / ### Pattern summary
_SEQ_TAGS_RE = re.compile(
    r"^\s*-\s*\*\*Sequence tags\*\*:\s*(.+?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_PATTERN_SUMMARY_RE = re.compile(
    r"^###\s*Pattern\s+summary\s*$\s*([\s\S]*?)(?=^###\s|^##\s|\Z)",
    re.MULTILINE | re.IGNORECASE,
)


def parse_kb_sequence_and_pattern(text: str) -> tuple[Optional[str], Optional[str]]:
    """
    從 KB 檔案前段文字擷取：
    - Sequence tags（行如 `- **Sequence tags**: lateral, logon`）
    - Pattern summary（### Pattern summary 底下以 `-` 開頭的條列，合併為短摘要）
    若無則回傳 (None, None)。
    """
    seq: Optional[str] = None
    m = _SEQ_TAGS_RE.search(text)
    if m:
        seq = m.group(1).strip()

    pat: Optional[str] = None
    m = _PATTERN_SUMMARY_RE.search(text)
    if m:
        block = m.group(1)
        bullets: list[str] = []
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("-"):
                bullets.append(line.lstrip("-").strip())
        if bullets:
            pat = " · ".join(bullets)
            if len(pat) > 480:
                pat = pat[:477].rstrip() + "…"

    return seq, pat


def normalize_kb_for_grouping(text: str) -> str:
    """
    將 KB 全文正規化以便「模板分組」：抹掉 episode 專屬 id、Hosts、Event count、Run、Entities 等，
    保留 **Sequence tags**、### Pattern summary 等語意，使同一模板、不同 window 的 cert 候選歸同一組。
    不含上述欄位的檔（如 SOP）幾乎不變，通常一檔一組。
    """
    t = text
    # 內文與標題中的 cert-…-w* episode id
    t = re.sub(r"cert-[A-Za-z0-9]+-w\d+", "EPISODE_ID", t)
    t = re.sub(r"(?m)^##\s+Episode\s*:.*$", "## Episode: EPISODE_ID", t)
    t = re.sub(r"(?m)^\s*-\s*\*\*Run\*\*:\s*.+$", "- **Run**: RUN_ID", t)
    t = re.sub(r"(?m)^\s*-\s*\*\*Entities \(users\)\*\*:\s*.+$", "- **Entities (users)**: USERS", t)
    t = re.sub(r"(?m)^\s*-\s*\*\*Hosts\*\*:\s*.+$", "- **Hosts**: HOSTS", t)
    t = re.sub(r"(?m)^\s*-\s*\*\*Event count\*\*:\s*.+$", "- **Event count**: N", t)
    return t.strip()


def group_key_from_text(text: str) -> str:
    """正規化後 SHA-256 前 12 hex，作為 template group id。"""
    n = normalize_kb_for_grouping(text)
    return hashlib.sha256(n.encode("utf-8")).hexdigest()[:12]


def _read_text_capped(path: Path, max_bytes: int = _MAX_BYTES_GROUPING) -> str:
    raw = path.read_bytes()
    if len(raw) > max_bytes:
        raw = raw[:max_bytes]
    return raw.decode("utf-8", errors="replace")


def kb_root_abs(repo_root: Path) -> Path:
    from src.config import get_config

    cfg = get_config()
    rel = cfg.KB_PATH.strip().strip("/").replace("\\", "/")
    if not rel:
        rel = "kb"
    return (repo_root / rel).resolve()


def _read_head(path: Path, max_chars: int = 48000) -> str:
    """只讀檔首，供擷取 front 區塊（避免超大檔）。"""
    with path.open(encoding="utf-8", errors="replace") as f:
        return f.read(max_chars)


def _is_kb_index_excluded(rel_posix: str) -> bool:
    """目錄／API 列舉時略過的檔名（例如根目錄說明檔，不當作知識條目）。"""
    return Path(rel_posix).name.lower() == "readme.md"


def list_kb_documents(repo_root: Path) -> list[dict[str, Any]]:
    """遞迴列出 KB 下所有 .md / .txt（或 KB_DB_MODE 下合併 DB），並附 sequence_tags、pattern_summary、group_key、group_size。"""
    from src.kb.loader import list_document_rows

    return list_document_rows(repo_root)


def validate_kb_relative_path(doc_path: str) -> str:
    """
    URL 路徑（不含前導 /）正規化為相對 KB 的 posix 路徑；不要求檔案已存在（DB-only 可用）。
    """
    if not doc_path or doc_path.strip() == "":
        raise HTTPException(status_code=400, detail="Missing path")
    parts = doc_path.replace("\\", "/").split("/")
    if ".." in parts or any(p.startswith("..") for p in parts):
        raise HTTPException(status_code=400, detail="Invalid path")
    rel = "/".join(parts)
    low = rel.lower()
    if not (low.endswith(".md") or low.endswith(".txt")):
        raise HTTPException(status_code=400, detail="Only .md and .txt are exposed")
    return rel


def _heuristic_summary_for_group(files: list[dict[str, Any]]) -> str:
    """規則型一句話摘要（繁中），無 tags/pattern 時用路徑與檔數。"""
    if not files:
        return ""
    f0 = files[0]
    tags = (f0.get("sequence_tags") or "").strip()
    pat = (f0.get("pattern_summary") or "").strip()
    rel = f0.get("rel_path", "")
    n = len(files)
    parts: list[str] = []
    if "cert_candidates_test" in rel:
        parts.append("測試用 CERT 候選")
    elif "cert_candidates" in rel:
        parts.append("CERT 時間窗候選片段")
    if tags:
        parts.append(f"序列標籤：{tags}")
    if pat:
        pshort = pat if len(pat) <= 220 else pat[:217] + "…"
        parts.append(pshort)
    if parts:
        return " · ".join(parts)
    top = rel.split("/")[0] if "/" in rel else rel
    if n > 1:
        return f"「{top}」路徑下共 {n} 份同模板 KB 文件"
    return f"「{rel}」（單檔知識庫）"


def aggregate_kb_groups(repo_root: Path) -> list[dict[str, Any]]:
    """
    將 list_kb_documents 平面結果依 group_key 聚合。
    每群含：group_key, group_size, files, heuristic_summary, representative_rel_path。
    """
    flat = list_kb_documents(repo_root)
    by_key: dict[str, list[dict[str, Any]]] = {}
    for r in flat:
        by_key.setdefault(r["group_key"], []).append(r)
    groups: list[dict[str, Any]] = []
    for gk in sorted(by_key.keys()):
        files = sorted(by_key[gk], key=lambda x: x["rel_path"])
        groups.append(
            {
                "group_key": gk,
                "group_size": len(files),
                "files": files,
                "heuristic_summary": _heuristic_summary_for_group(files),
                "representative_rel_path": files[0]["rel_path"],
            }
        )
    return groups


def resolve_kb_file(repo_root: Path, doc_path: str) -> Path:
    """
    doc_path：URL path，如 cert_candidates/foo.md（不含前導 /）。
    回傳絕對路徑；若不在 KB 根下或不是檔案則 HTTPException。
    """
    if not doc_path or doc_path.strip() == "":
        raise HTTPException(status_code=400, detail="Missing path")
    # 正規化：拒絕 .. 與絕對
    parts = doc_path.replace("\\", "/").split("/")
    if ".." in parts or any(p.startswith("..") for p in parts):
        raise HTTPException(status_code=400, detail="Invalid path")

    kb = kb_root_abs(repo_root)
    if not kb.is_dir():
        raise HTTPException(status_code=404, detail="KB root not found")

    rel = "/".join(parts)
    target = (kb / rel).resolve()
    try:
        target.relative_to(kb)
    except ValueError:
        raise HTTPException(status_code=400, detail="Path outside KB") from None
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Document not found")
    if target.suffix.lower() not in (".md", ".txt"):
        raise HTTPException(status_code=400, detail="Only .md and .txt are exposed")
    return target


def read_kb_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def escape_html_body(text: str) -> str:
    return html.escape(text, quote=True)
