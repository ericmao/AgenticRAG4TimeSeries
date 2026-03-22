"""
MVP Web UI：analysis_runs 列表／詳情、產物 JSON、觸發 CERT Layer C 執行。

環境變數:
  DATABASE_URL
  REPO_ROOT（可選，預設為此檔案上兩層目錄的 repo 根）
  KB_PATH（可選，預設 kb/，相對 REPO_ROOT；KB 瀏覽 /kb 用）
  KB_DB_MODE（可選：off | merge | db_only；非 off 時需 DATABASE_URL，寫入需 MVP_UI_API_KEY）
  MVP_UI_API_KEY（可選；若設定則 POST /api/runs/cert 需 Header X-API-Key）

啟動:
  cd AgenticRAG4TimeSeries && PYTHONPATH=. uvicorn services.mvp_ui_api.app:app --host 0.0.0.0 --port 8765
"""
from __future__ import annotations

import json as json_lib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from markupsafe import Markup
from pydantic import BaseModel, Field

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.storage.analysis_runs_store import get_adjacent_run_ids, get_run_by_id, list_runs

from .kb_browser import (
    aggregate_kb_groups,
    escape_html_body,
    kb_root_abs,
    list_kb_documents,
    validate_kb_relative_path,
)
from .kb_file_llm import enrich_files_llm_summaries
from .kb_group_llm import enrich_groups_llm_summaries

REPO_ROOT = Path(os.environ.get("REPO_ROOT", str(_REPO))).resolve()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _tojson_filter(obj: Any, indent: Optional[int] = None, **kwargs: Any) -> Markup:
    # Jinja: {{ x | tojson(indent=2) }} passes indent as keyword
    _ = kwargs  # ignore extras for forward compatibility
    return Markup(json_lib.dumps(obj, indent=indent, ensure_ascii=False, default=str))


templates.env.filters["tojson"] = _tojson_filter

app = FastAPI(title="AgenticRAG MVP UI", version="0.1.0")


def _db_url() -> str:
    u = os.environ.get("DATABASE_URL", "").strip()
    if not u:
        raise HTTPException(status_code=503, detail="DATABASE_URL not set")
    return u


def _runs_row_display(row: dict[str, Any]) -> dict[str, Any]:
    """Template 用：created_at ISO、保留 layerc_summary。"""
    r = dict(row)
    ca = r.get("created_at")
    if ca is not None and hasattr(ca, "isoformat"):
        r["created_at_iso"] = ca.isoformat()
    else:
        r["created_at_iso"] = str(ca) if ca is not None else ""
    return r


def _check_api_key(request: Request) -> None:
    key = os.environ.get("MVP_UI_API_KEY", "").strip()
    if not key:
        return
    if request.headers.get("x-api-key") != key:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


class CertRunBody(BaseModel):
    episode_path: str = Field(..., description="Relative to repo root, e.g. tests/demo/episode_insider_highrisk.json")
    dataset_label: str = "cert"
    skip_db: bool = False
    no_writeback: bool = False
    writeback_mode: str = "dry_run"
    triage_rules: Optional[list[str]] = Field(
        default=None,
        description="Optional rule_id list; omitted uses TRIAGE_RULES env or episode.sequence_tags",
    )


def _artifact_path(episode_id: str, kind: str) -> Path:
    safe_kinds = {"evidence", "writeback", "issues"}
    if kind not in safe_kinds:
        raise HTTPException(status_code=400, detail="kind must be evidence|writeback|issues")
    sub = {"evidence": "evidence", "writeback": "writeback", "issues": "issues"}[kind]
    base = REPO_ROOT / "outputs" / sub
    path = base / f"{episode_id}.json"
    try:
        path.resolve().relative_to((REPO_ROOT / "outputs").resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid path") from None
    return path


def _kb_path_display() -> str:
    root = kb_root_abs(REPO_ROOT)
    try:
        return root.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(root)


def _prepare_kb_groups() -> list[dict[str, Any]]:
    groups = aggregate_kb_groups(REPO_ROOT)
    try:
        enrich_groups_llm_summaries(REPO_ROOT, groups)
    except Exception:
        for g in groups:
            g.setdefault("llm_summary", "")
    try:
        enrich_files_llm_summaries(REPO_ROOT, groups)
    except Exception:
        for g in groups:
            for f in g.get("files") or []:
                f.setdefault("llm_file_summary", "")
    for g in groups:
        llm = (g.get("llm_summary") or "").strip()
        g["display_summary"] = llm or (g.get("heuristic_summary") or "")
        for f in g.get("files") or []:
            f["href_path"] = quote(f["rel_path"], safe="")
            f["href_edit"] = quote(f["rel_path"], safe="")
    return groups


def _kb_db_enabled() -> bool:
    from src.kb.loader import get_kb_db_mode

    return get_kb_db_mode() != "off"


@app.get("/kb", response_class=HTMLResponse)
def kb_list_page(request: Request):
    try:
        groups = _prepare_kb_groups()
    except Exception as e:
        return templates.TemplateResponse(
            "kb_list.html",
            {
                "request": request,
                "groups": [],
                "error": str(e),
                "kb_root_display": _kb_path_display(),
                "kb_db_enabled": _kb_db_enabled(),
            },
        )
    return templates.TemplateResponse(
        "kb_list.html",
        {
            "request": request,
            "groups": groups,
            "error": None,
            "kb_root_display": _kb_path_display(),
            "kb_db_enabled": _kb_db_enabled(),
        },
    )


@app.get("/kb/p/{doc_path:path}", response_class=HTMLResponse)
def kb_detail_page(request: Request, doc_path: str):
    from src.kb.loader import read_kb_body

    rel = validate_kb_relative_path(doc_path)
    try:
        text = read_kb_body(REPO_ROOT, rel)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Document not found")
    size = len(text.encode("utf-8", errors="replace"))
    return templates.TemplateResponse(
        "kb_detail.html",
        {
            "request": request,
            "title": Path(rel).name,
            "rel_path": rel,
            "size": size,
            "body_html": Markup(escape_html_body(text)),
            "kb_db_enabled": _kb_db_enabled(),
            "href_edit": quote(rel, safe=""),
        },
    )


@app.get("/api/kb/files")
def api_kb_files():
    return list_kb_documents(REPO_ROOT)


@app.get("/api/kb/groups")
def api_kb_groups():
    """依 group_key 聚合；llm_summary／llm_file_summary 僅來自快取檔（瀏覽時不呼叫 LLM）。"""
    groups = aggregate_kb_groups(REPO_ROOT)
    try:
        enrich_groups_llm_summaries(REPO_ROOT, groups)
    except Exception:
        for g in groups:
            g.setdefault("llm_summary", "")
    try:
        enrich_files_llm_summaries(REPO_ROOT, groups)
    except Exception:
        for g in groups:
            for f in g.get("files") or []:
                f.setdefault("llm_file_summary", "")
    for g in groups:
        llm = (g.get("llm_summary") or "").strip()
        g["display_summary"] = llm or (g.get("heuristic_summary") or "")
    return {"groups": groups}


@app.get("/api/kb/raw/{doc_path:path}")
def api_kb_raw(doc_path: str):
    from src.kb.loader import read_kb_body

    rel = validate_kb_relative_path(doc_path)
    try:
        text = read_kb_body(REPO_ROOT, rel)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Document not found")
    return PlainTextResponse(text, media_type="text/plain; charset=utf-8")


class KbDocumentPutBody(BaseModel):
    body: str = Field(..., description="Full document text")
    expected_version: int = Field(..., description="0 for new overlay; else current kb_documents.version")
    editor: Optional[str] = None
    note: Optional[str] = None


@app.get("/api/kb/documents/{doc_path:path}")
def api_kb_document_meta(doc_path: str):
    """目前合併後內容與 DB 版本號（無 DB 列時 version=0）。"""
    from src.kb.loader import get_kb_db_mode, read_kb_body

    if get_kb_db_mode() == "off":
        raise HTTPException(status_code=404, detail="KB_DB_MODE is off")
    rel = validate_kb_relative_path(doc_path)
    try:
        text = read_kb_body(REPO_ROOT, rel)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Document not found")
    rec = None
    try:
        db_url = _db_url()
        from src.storage.kb_documents_store import get_kb_document_by_path

        rec = get_kb_document_by_path(db_url, rel)
    except HTTPException:
        raise
    except Exception:
        pass
    return {
        "rel_path": rel,
        "body": text,
        "version": int(rec["version"]) if rec else 0,
        "updated_at": rec.get("updated_at").isoformat() if rec and rec.get("updated_at") and hasattr(rec["updated_at"], "isoformat") else None,
        "in_database": rec is not None,
    }


@app.get("/api/kb/documents/{doc_path:path}/versions")
def api_kb_document_versions(doc_path: str, limit: int = 50):
    from src.kb.loader import get_kb_db_mode
    from src.storage.kb_documents_store import list_kb_document_versions

    if get_kb_db_mode() == "off":
        raise HTTPException(status_code=404, detail="KB_DB_MODE is off")
    rel = validate_kb_relative_path(doc_path)
    try:
        rows = list_kb_document_versions(_db_url(), rel, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return {"rel_path": rel, "versions": rows}


@app.put("/api/kb/documents/{doc_path:path}")
def api_kb_document_put(request: Request, doc_path: str, payload: KbDocumentPutBody):
    from src.kb.loader import get_kb_db_mode

    if get_kb_db_mode() == "off":
        raise HTTPException(status_code=400, detail="KB_DB_MODE is off")
    _check_api_key(request)
    rel = validate_kb_relative_path(doc_path)
    from src.storage.kb_documents_store import KbVersionConflict, upsert_kb_document

    try:
        out = upsert_kb_document(
            _db_url(),
            rel,
            payload.body,
            payload.expected_version,
            editor=payload.editor,
            note=payload.note,
        )
    except KbVersionConflict as ex:
        raise HTTPException(
            status_code=409,
            detail={"message": "version conflict", "current_version": ex.current_version},
        ) from ex
    return out


@app.get("/kb/e/{doc_path:path}", response_class=HTMLResponse)
def kb_edit_page(request: Request, doc_path: str):
    from src.kb.loader import get_kb_db_mode, read_kb_body
    from src.storage.kb_documents_store import get_kb_document_by_path

    if get_kb_db_mode() == "off":
        raise HTTPException(status_code=404, detail="KB editing requires KB_DB_MODE merge or db_only")
    rel = validate_kb_relative_path(doc_path)
    try:
        text = read_kb_body(REPO_ROOT, rel)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Document not found")
    ver = 0
    try:
        rec = get_kb_document_by_path(_db_url(), rel)
        if rec:
            ver = int(rec["version"])
    except HTTPException:
        raise
    except Exception:
        pass
    return templates.TemplateResponse(
        "kb_edit.html",
        {
            "request": request,
            "rel_path": rel,
            "href_path": quote(rel, safe=""),
            "body_text": text,
            "version": ver,
        },
    )


@app.get("/health")
def health():
    return {"status": "ok", "service": "mvp_ui_api"}


@app.get("/", response_class=RedirectResponse)
def root():
    return RedirectResponse(url="/runs", status_code=302)


@app.get("/runs", response_class=HTMLResponse)
def runs_page(request: Request, source: Optional[str] = None):
    try:
        raw = list_runs(_db_url(), limit=100, offset=0, source=source)
        rows = [_runs_row_display(r) for r in raw]
    except Exception as e:
        rows = []
        err = str(e)
        return templates.TemplateResponse(
            "runs.html",
            {"request": request, "runs": [], "error": err, "source": source or ""},
        )
    return templates.TemplateResponse(
        "runs.html",
        {"request": request, "runs": rows, "error": None, "source": source or ""},
    )


@app.get("/runs/{run_id}", response_class=HTMLResponse)
def run_detail_page(request: Request, run_id: int):
    row = get_run_by_id(_db_url(), run_id)
    if not row:
        return templates.TemplateResponse(
            "run_not_found.html",
            {"request": request, "run_id": run_id},
            status_code=404,
        )
    # datetime etc. JSON-serializable for display
    display = {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in row.items()}
    ao = row.get("agent_outputs_json") or {}
    by_rule = ao.get("by_rule") if isinstance(ao, dict) else None
    nav_newer_id, nav_older_id = get_adjacent_run_ids(_db_url(), run_id)
    return templates.TemplateResponse(
        "run_detail.html",
        {
            "request": request,
            "run": display,
            "episode_id": row.get("episode_id"),
            "by_rule": by_rule,
            "nav_newer_id": nav_newer_id,
            "nav_older_id": nav_older_id,
        },
    )


@app.get("/api/runs")
def api_runs(source: Optional[str] = None):
    return list_runs(_db_url(), limit=200, offset=0, source=source)


@app.get("/api/runs/{run_id}")
def api_run(run_id: int, summary: Optional[int] = None):
    row = get_run_by_id(_db_url(), run_id)
    if not row:
        raise HTTPException(status_code=404)
    out: dict[str, Any] = {}
    for k, v in row.items():
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()
        else:
            out[k] = v
    if summary:
        ao = out.get("agent_outputs_json") or {}
        br = ao.get("by_rule") if isinstance(ao, dict) else None
        keys = list(br.keys()) if isinstance(br, dict) else []
        return {
            "id": out.get("id"),
            "created_at": out.get("created_at"),
            "episode_id": out.get("episode_id"),
            "status": out.get("status"),
            "source": out.get("source"),
            "layerc_summary": out.get("layerc_summary"),
            "by_rule_keys": keys,
        }
    return out


@app.get("/api/artifacts/{episode_id}/{kind}")
def api_artifact(episode_id: str, kind: str):
    path = _artifact_path(episode_id, kind)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path, media_type="application/json", filename=path.name)


def _background_cert(body: CertRunBody) -> None:
    ep = (REPO_ROOT / body.episode_path).resolve()
    try:
        ep.relative_to(REPO_ROOT)
    except ValueError:
        return
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/mvp_cert_layer_c.py"),
        "--episode",
        str(ep.relative_to(REPO_ROOT)),
        "--dataset-label",
        body.dataset_label,
        "--writeback-mode",
        body.writeback_mode,
    ]
    if body.skip_db:
        cmd.append("--skip-db")
    if body.no_writeback:
        cmd.append("--no-writeback")
    if body.triage_rules:
        cmd.extend(["--triage-rules", ",".join(body.triage_rules)])
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=False)


@app.post("/api/runs/cert")
def api_run_cert(request: Request, body: CertRunBody, background_tasks: BackgroundTasks):
    _check_api_key(request)
    ep = (REPO_ROOT / body.episode_path).resolve()
    if not ep.is_file():
        raise HTTPException(status_code=400, detail=f"Episode file not found: {body.episode_path}")
    try:
        ep.relative_to(REPO_ROOT)
    except ValueError:
        raise HTTPException(status_code=400, detail="Path must be under repo root") from None
    background_tasks.add_task(_background_cert, body)
    return {"ok": True, "message": "cert pipeline started in background", "episode_path": body.episode_path}
