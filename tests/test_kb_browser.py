"""KB 瀏覽：列檔與路徑安全。"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from services.mvp_ui_api import kb_browser


@pytest.fixture(autouse=True)
def _force_kb_db_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """避免本機 .env 的 KB_DB_MODE 影響純檔案列舉測試；並清掉 DATABASE_URL 以免誤連真實庫。"""
    monkeypatch.setenv("KB_DB_MODE", "off")
    monkeypatch.delenv("DATABASE_URL", raising=False)


def test_resolve_rejects_parent_segments(tmp_path: Path):
    with pytest.raises(HTTPException) as exc:
        kb_browser.resolve_kb_file(tmp_path, "a/../../etc/passwd")
    assert exc.value.status_code == 400


def test_list_kb_documents(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "top.md").write_text("# t", encoding="utf-8")
    (kb / "README.md").write_text("# readme", encoding="utf-8")
    sub = kb / "sub"
    sub.mkdir()
    (sub / "nested.md").write_text("x", encoding="utf-8")

    monkeypatch.setattr(kb_browser, "kb_root_abs", lambda _repo: kb)

    rows = kb_browser.list_kb_documents(tmp_path)
    rels = sorted(r["rel_path"] for r in rows)
    assert rels == ["sub/nested.md", "top.md"]
    assert "README.md" not in rels
    for r in rows:
        assert "sequence_tags" in r and "pattern_summary" in r


def test_normalize_groups_cert_windows_with_same_template() -> None:
    a = Path("kb/cert_candidates/cert-AAB0754-w13.md")
    b = Path("kb/cert_candidates/cert-AAB0754-w23.md")
    if not a.is_file() or not b.is_file():
        pytest.skip("fixture kb cert files missing")
    ta = a.read_text(encoding="utf-8")
    tb = b.read_text(encoding="utf-8")
    assert kb_browser.group_key_from_text(ta) == kb_browser.group_key_from_text(tb)


def test_group_key_differs_for_different_sop_files() -> None:
    p1 = Path("kb/sop_insider_anomaly.md")
    p2 = Path("kb/hunt_query_templates.md")
    if not p1.is_file() or not p2.is_file():
        pytest.skip("kb fixtures missing")
    assert kb_browser.group_key_from_text(p1.read_text(encoding="utf-8")) != kb_browser.group_key_from_text(
        p2.read_text(encoding="utf-8")
    )


def test_parse_sequence_and_pattern_from_cert_style_md() -> None:
    md = """## Episode: x

- **Run**: r1
- **Sequence tags**: lateral, logoff, logon
- **Entities (users)**: U1

### Pattern summary

- First bullet line.
- Second bullet line.
"""
    seq, pat = kb_browser.parse_kb_sequence_and_pattern(md)
    assert seq == "lateral, logoff, logon"
    assert "First bullet" in (pat or "")
    assert "Second bullet" in (pat or "")


def test_aggregate_kb_groups_sizes_and_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    kb = tmp_path / "kb"
    kb.mkdir()
    same_body = """## Episode: cert-X-w13
- **Sequence tags**: lateral
### Pattern summary
- A
"""
    (kb / "a.md").write_text(same_body, encoding="utf-8")
    (kb / "b.md").write_text(same_body.replace("w13", "w23"), encoding="utf-8")
    (kb / "other.md").write_text("# unique content for solo group\n", encoding="utf-8")
    monkeypatch.setattr(kb_browser, "kb_root_abs", lambda _repo: kb)

    groups = kb_browser.aggregate_kb_groups(tmp_path)
    assert len(groups) == 2
    by_size = sorted(g["group_size"] for g in groups)
    assert by_size == [1, 2]
    two = next(g for g in groups if g["group_size"] == 2)
    assert len(two["files"]) == 2
    assert two["heuristic_summary"]
    assert two["representative_rel_path"] in ("a.md", "b.md")


def test_enrich_group_llm_from_cache_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import json

    from services.mvp_ui_api import kb_group_llm

    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "solo.md").write_text("hello world unique\n", encoding="utf-8")
    monkeypatch.setattr(kb_browser, "kb_root_abs", lambda _repo: kb)

    groups = kb_browser.aggregate_kb_groups(tmp_path)
    gk = groups[0]["group_key"]
    out = tmp_path / "outputs"
    out.mkdir()
    (out / ".kb_group_llm_cache.json").write_text(
        json.dumps({gk: {"summary": "快取群組摘要"}}, ensure_ascii=False),
        encoding="utf-8",
    )

    kb_group_llm.enrich_groups_llm_summaries(tmp_path, groups)
    assert groups[0].get("llm_summary") == "快取群組摘要"


def test_enrich_group_llm_structured_without_legacy_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import json

    from services.mvp_ui_api import kb_group_llm

    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "solo.md").write_text("hello world unique\n", encoding="utf-8")
    monkeypatch.setattr(kb_browser, "kb_root_abs", lambda _repo: kb)

    groups = kb_browser.aggregate_kb_groups(tmp_path)
    gk = groups[0]["group_key"]
    out = tmp_path / "outputs"
    out.mkdir()
    (out / ".kb_group_llm_cache.json").write_text(
        json.dumps(
            {
                gk: {
                    "meaning": "意義內容",
                    "usage_direction": "使用方向",
                    "threats": "威脅對應",
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    kb_group_llm.enrich_groups_llm_summaries(tmp_path, groups)
    assert groups[0].get("llm_meaning") == "意義內容"
    assert groups[0].get("llm_usage_direction") == "使用方向"
    assert groups[0].get("llm_threats") == "威脅對應"
    assert groups[0].get("llm_summary") == "意義內容"


def test_enrich_group_llm_db_overrides_json_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import json

    from services.mvp_ui_api import kb_group_llm

    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "solo.md").write_text("hello world unique\n", encoding="utf-8")
    monkeypatch.setattr(kb_browser, "kb_root_abs", lambda _repo: kb)

    groups = kb_browser.aggregate_kb_groups(tmp_path)
    gk = groups[0]["group_key"]
    out = tmp_path / "outputs"
    out.mkdir()
    (out / ".kb_group_llm_cache.json").write_text(
        json.dumps({gk: {"meaning": "json", "usage_direction": "", "threats": "", "summary": ""}}, ensure_ascii=False),
        encoding="utf-8",
    )

    monkeypatch.setenv("DATABASE_URL", "postgresql://unused:unused@127.0.0.1:1/agentic")
    monkeypatch.setattr(
        "src.storage.kb_group_llm_store.fetch_all_kb_group_llm",
        lambda _url: {
            gk: {
                "meaning": "db意義",
                "usage_direction": "db方向",
                "threats": "db威脅",
                "summary": "",
                "content_fingerprint": None,
                "model": None,
                "updated_at": None,
            }
        },
    )

    kb_group_llm.enrich_groups_llm_summaries(tmp_path, groups)
    assert groups[0].get("llm_meaning") == "db意義"
    assert groups[0].get("llm_usage_direction") == "db方向"
    assert groups[0].get("llm_threats") == "db威脅"


def test_enrich_file_llm_from_cache_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import hashlib
    import json

    from services.mvp_ui_api import kb_file_llm

    kb = tmp_path / "kb"
    kb.mkdir()
    text = "# title\nbody\n"
    (kb / "a.md").write_text(text, encoding="utf-8")
    monkeypatch.setattr(kb_browser, "kb_root_abs", lambda _repo: kb)
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    out = tmp_path / "outputs"
    out.mkdir()
    (out / ".kb_file_llm_cache.json").write_text(
        json.dumps(
            {"a.md": {"summary": "快取單檔概要", "content_sha256": sha}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    groups = kb_browser.aggregate_kb_groups(tmp_path)
    kb_file_llm.enrich_files_llm_summaries(tmp_path, groups)
    assert groups[0]["files"][0].get("llm_file_summary") == "快取單檔概要"


def test_read_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "x.md").write_text("hello 世界", encoding="utf-8")
    monkeypatch.setattr(kb_browser, "kb_root_abs", lambda _repo: kb)

    p = kb_browser.resolve_kb_file(tmp_path, "x.md")
    assert kb_browser.read_kb_text(p) == "hello 世界"


def test_truncate_llm_file_summary_for_display() -> None:
    from services.mvp_ui_api.kb_file_llm import truncate_llm_file_summary_for_display

    assert truncate_llm_file_summary_for_display("") == ""
    assert truncate_llm_file_summary_for_display("  hello  \n") == "hello"
    long = " ".join(f"w{i}" for i in range(250))
    out = truncate_llm_file_summary_for_display(long, max_words=200)
    assert out.endswith("…")
    assert len(out.split()) == 200
    big_zh = "字" * 2000
    out2 = truncate_llm_file_summary_for_display(big_zh, max_chars=100)
    assert len(out2) <= 101
    assert out2.endswith("…")


def test_kb_describer_runner_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.layer_c.agents.kb_describer_agent import kb_describer_runner

    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "x.md").write_text("# t\nbody\n", encoding="utf-8")
    monkeypatch.setattr(kb_browser, "kb_root_abs", lambda _repo: kb)

    out = kb_describer_runner({"repo_root": str(tmp_path), "dry_run": True})
    assert out["ok"] is True
    assert out["stats"]["dry_run_would"] >= 1
