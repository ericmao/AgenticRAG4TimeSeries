"""
MVP KB retriever: load *.md/*.txt under KB_PATH, chunk, keyword-score, return EvidenceItem candidates.
No vector DB; deterministic.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Optional

from src.contracts.evidence import EvidenceItem
from src.contracts.episode import Episode, Hypothesis


def _stable_evidence_id(
    source: str,
    kind: str,
    title: str,
    body: str,
    stix_id: Optional[str] = None,
    chunk_id: Optional[str] = None,
) -> str:
    """Stable hash for evidence_id and deduplication."""
    payload = f"{source}|{kind}|{title}|{body}|{stix_id or ''}|{chunk_id or ''}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _chunk_text(text: str, size: int = 700, overlap: int = 120) -> list[str]:
    """Split text into overlapping chunks; deterministic order."""
    if not text or size <= 0:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
        if start <= 0:
            start = end
    return chunks


def _score_chunk(chunk: str, query_strings: list[str]) -> float:
    """Simple keyword match: fraction of query terms that appear in chunk (case-insensitive). Capped 0-1."""
    if not query_strings:
        return 0.0
    chunk_lower = chunk.lower()
    hits = sum(1 for q in query_strings if q.strip() and q.strip().lower() in chunk_lower)
    return min(1.0, hits / len(query_strings)) if query_strings else 0.0


def _load_kb_docs(kb_path: Path) -> list[tuple[str, str]]:
    """Load all *.md and *.txt under kb_path. Return list of (filename, content)."""
    if not kb_path.is_dir():
        return []
    out: list[tuple[str, str]] = []
    for ext in ("*.md", "*.txt"):
        for f in sorted(kb_path.rglob(ext)):
            if f.is_file():
                try:
                    content = f.read_text(encoding="utf-8", errors="replace")
                    name = str(f.relative_to(kb_path))
                    out.append((name, content))
                except Exception:
                    continue
    return sorted(out, key=lambda x: x[0])


def retrieve_from_kb(
    episode: Episode,
    query_strings: list[str],
    hypothesis: Optional[Hypothesis] = None,
    kb_path: Optional[Path] = None,
    chunk_size: int = 700,
    chunk_overlap: int = 120,
) -> list[EvidenceItem]:
    """
    Load docs from KB_PATH, chunk, score by keyword match, return EvidenceItem candidates.
    source="kb", kind="snippet", provenance includes query and retrieved_at_ms.
    """
    if kb_path is None:
        from src.config import get_config
        cfg = get_config()
        repo_root = Path(__file__).resolve().parents[2]
        kb_path = repo_root / cfg.KB_PATH.strip().lstrip("/")
    retrieved_at_ms = int(time.time() * 1000)
    docs = _load_kb_docs(kb_path)
    candidates: list[EvidenceItem] = []
    for filename, content in docs:
        chunks = _chunk_text(content, size=chunk_size, overlap=chunk_overlap)
        for i, chunk in enumerate(chunks):
            score = _score_chunk(chunk, query_strings)
            chunk_id = f"{filename}#{i}"
            evidence_id = _stable_evidence_id("kb", "snippet", filename, chunk, None, chunk_id)
            prov = {
                "retrieved_at_ms": retrieved_at_ms,
                "query": " ".join(query_strings[:5]) if query_strings else "",
            }
            candidates.append(
                EvidenceItem(
                    evidence_id=evidence_id,
                    source="kb",
                    kind="snippet",
                    title=filename,
                    body=chunk,
                    stix_id=None,
                    chunk_id=chunk_id,
                    score=round(score, 4),
                    ts_ms=None,
                    provenance=prov,
                )
            )
    return candidates
