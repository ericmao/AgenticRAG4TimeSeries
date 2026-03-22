"""KB loader：路徑驗證與 KB_DB_MODE=off 行為。"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from services.mvp_ui_api import kb_browser
from src.kb.loader import list_document_rows, read_kb_body


@pytest.fixture(autouse=True)
def _kb_db_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KB_DB_MODE", "off")


def test_validate_kb_relative_path_ok() -> None:
    assert kb_browser.validate_kb_relative_path("foo/bar.md") == "foo/bar.md"


def test_validate_kb_rejects_invalid() -> None:
    with pytest.raises(HTTPException) as exc:
        kb_browser.validate_kb_relative_path("a/../x.md")
    assert exc.value.status_code == 400


def test_list_document_rows_respects_monkeypatch_kb_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "only.md").write_text("hello", encoding="utf-8")
    monkeypatch.setattr(kb_browser, "kb_root_abs", lambda _repo: kb)

    rows = list_document_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["rel_path"] == "only.md"


def test_read_kb_body_filesystem(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "x.md").write_text("content", encoding="utf-8")
    monkeypatch.setattr(kb_browser, "kb_root_abs", lambda _repo: kb)

    assert read_kb_body(tmp_path, "x.md") == "content"
