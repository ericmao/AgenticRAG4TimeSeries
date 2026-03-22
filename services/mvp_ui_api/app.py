"""
MVP Web UI：analysis_runs 列表／詳情、產物 JSON、觸發 CERT Layer C 執行。

環境變數:
  DATABASE_URL
  REPO_ROOT（可選，預設為此檔案上兩層目錄的 repo 根）
  KB_PATH（可選，預設 kb/，相對 REPO_ROOT；KB 瀏覽 /kb 用）
  KB_DB_MODE（可選：off | merge | db_only；非 off 時需 DATABASE_URL，寫入需 MVP_UI_API_KEY）
  MVP_UI_API_KEY（可選；若設定則 POST /api/runs/cert、/api/runs/wazuh 需 Header X-API-Key）

啟動:
  cd AgenticRAG4TimeSeries && PYTHONPATH=. uvicorn services.mvp_ui_api.app:app --host 0.0.0.0 --port 8765
"""
from __future__ import annotations

import asyncio
import json as json_lib
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup
from pydantic import BaseModel, Field, field_validator

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.storage.analysis_runs_store import (
    get_adjacent_run_ids,
    get_run_by_episode_id,
    get_run_by_id,
    list_runs,
)

from .kb_browser import (
    aggregate_kb_groups,
    escape_html_body,
    kb_root_abs,
    list_kb_documents,
    validate_kb_relative_path,
)
from .kb_file_llm import enrich_files_llm_summaries, truncate_llm_file_summary_for_display
from .kb_group_llm import enrich_groups_llm_summaries
from .agent_dashboard import AGENT_CATALOG, read_agent_activity
from .pipeline_jobs import finalize_job, read_job, validate_job_id, write_job_running

REPO_ROOT = Path(os.environ.get("REPO_ROOT", str(_REPO))).resolve()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
INVESTIGATION_STATIC = Path(__file__).resolve().parent / "static" / "investigation"


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


class WazuhRunBody(BaseModel):
    """背景執行 mvp_wazuh_episode_pg.py（與 CLI 一致；macOS／Windows 皆使用目前 Python 直譯器）。"""

    target_ip: str = Field(..., min_length=1, description="監視 IP，例如 10.0.0.5")
    match_all: bool = False
    hours: int = Field(24, ge=1, le=168)
    evidenceops: bool = False
    no_writeback: bool = False
    writeback_mode: str = "dry_run"
    skip_db: bool = False
    triage_rules: Optional[list[str]] = Field(
        default=None,
        description="Optional triage rule_id list",
    )

    @field_validator("target_ip")
    @classmethod
    def _target_ip_safe(cls, v: str) -> str:
        s = (v or "").strip()
        if not s or ".." in s or "/" in s or "\\" in s:
            raise ValueError("invalid target_ip")
        return s


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


def _kb_display_summary_for_group(g: dict[str, Any]) -> str:
    """列表一行摘要：legacy llm_summary 優先，否則取 meaning 前段，再退回啟發式。"""
    llm = (g.get("llm_summary") or "").strip()
    if llm:
        return llm
    m = (g.get("llm_meaning") or "").strip()
    if m:
        return m[:180] + ("…" if len(m) > 180 else "")
    return (g.get("heuristic_summary") or "")


def _prepare_kb_groups() -> list[dict[str, Any]]:
    groups = aggregate_kb_groups(REPO_ROOT)
    try:
        enrich_groups_llm_summaries(REPO_ROOT, groups)
    except Exception:
        for g in groups:
            g.setdefault("llm_summary", "")
            g.setdefault("llm_meaning", "")
            g.setdefault("llm_usage_direction", "")
            g.setdefault("llm_threats", "")
    try:
        enrich_files_llm_summaries(REPO_ROOT, groups)
    except Exception:
        for g in groups:
            for f in g.get("files") or []:
                f.setdefault("llm_file_summary", "")
    for g in groups:
        g["display_summary"] = _kb_display_summary_for_group(g)
        for f in g.get("files") or []:
            f["href_path"] = quote(f["rel_path"], safe="")
            f["href_edit"] = quote(f["rel_path"], safe="")
            f["llm_file_description"] = truncate_llm_file_summary_for_display(
                f.get("llm_file_summary") or "",
            )
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
    """依 group_key 聚合；llm_* 僅來自快取檔（瀏覽時不呼叫 LLM）。"""
    groups = aggregate_kb_groups(REPO_ROOT)
    try:
        enrich_groups_llm_summaries(REPO_ROOT, groups)
    except Exception:
        for g in groups:
            g.setdefault("llm_summary", "")
            g.setdefault("llm_meaning", "")
            g.setdefault("llm_usage_direction", "")
            g.setdefault("llm_threats", "")
    try:
        enrich_files_llm_summaries(REPO_ROOT, groups)
    except Exception:
        for g in groups:
            for f in g.get("files") or []:
                f.setdefault("llm_file_summary", "")
    for g in groups:
        g["display_summary"] = _kb_display_summary_for_group(g)
        for f in g.get("files") or []:
            f["llm_file_description"] = truncate_llm_file_summary_for_display(
                f.get("llm_file_summary") or "",
            )
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


@app.get("/agents", response_class=HTMLResponse)
def agents_page(request: Request):
    snap = read_agent_activity(REPO_ROOT)
    return templates.TemplateResponse(
        "agents.html",
        {
            "request": request,
            "catalog": AGENT_CATALOG,
            "activity": snap,
        },
    )


@app.get("/api/agents/catalog")
def api_agents_catalog():
    return {"agents": AGENT_CATALOG}


@app.get("/api/agents/status")
def api_agents_status():
    return read_agent_activity(REPO_ROOT)


@app.get("/api/agents/stream")
async def api_agents_stream():
    """SSE: push agent_activity.json snapshot every second (for cross-process orchestrator updates)."""

    async def gen():
        while True:
            payload = read_agent_activity(REPO_ROOT)
            line = "data: " + json_lib.dumps(payload, ensure_ascii=False) + "\n\n"
            yield line.encode("utf-8")
            await asyncio.sleep(1)

    return StreamingResponse(gen(), media_type="text/event-stream")


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


def _wazuh_request_dict(body: WazuhRunBody) -> dict[str, Any]:
    return {
        "target_ip": body.target_ip,
        "hours": body.hours,
        "match_all": body.match_all,
        "evidenceops": body.evidenceops,
        "no_writeback": body.no_writeback,
        "skip_db": body.skip_db,
        "writeback_mode": body.writeback_mode,
        "triage_rules": body.triage_rules,
    }


def _background_wazuh(body: WazuhRunBody, job_id: str) -> None:
    req = _wazuh_request_dict(body)
    script = REPO_ROOT / "scripts" / "mvp_wazuh_episode_pg.py"
    cmd = [
        sys.executable,
        str(script),
        "--target-ip",
        body.target_ip,
        "--hours",
        str(body.hours),
        "--writeback-mode",
        body.writeback_mode,
    ]
    if body.match_all:
        cmd.append("--match-all")
    if body.evidenceops:
        cmd.append("--evidenceops")
    if body.no_writeback:
        cmd.append("--no-writeback")
    if body.skip_db:
        cmd.append("--skip-db")
    if body.triage_rules:
        cmd.extend(["--triage-rules", ",".join(body.triage_rules)])
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    try:
        r = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=7200,
        )
        finalize_job(
            REPO_ROOT,
            job_id,
            kind="wazuh",
            request=req,
            exit_code=r.returncode,
            stdout=r.stdout or "",
            stderr=r.stderr or "",
        )
    except subprocess.TimeoutExpired as e:
        finalize_job(
            REPO_ROOT,
            job_id,
            kind="wazuh",
            request=req,
            exit_code=-1,
            stderr=str(e),
            error="subprocess timeout (7200s)",
        )
    except Exception as e:
        finalize_job(
            REPO_ROOT,
            job_id,
            kind="wazuh",
            request=req,
            exit_code=-1,
            error=str(e)[:4000],
        )


def _cert_request_dict(body: CertRunBody) -> dict[str, Any]:
    return {
        "episode_path": body.episode_path,
        "dataset_label": body.dataset_label,
        "skip_db": body.skip_db,
        "no_writeback": body.no_writeback,
        "writeback_mode": body.writeback_mode,
        "triage_rules": body.triage_rules,
    }


def _background_cert(body: CertRunBody, job_id: str) -> None:
    req = _cert_request_dict(body)
    ep = (REPO_ROOT / body.episode_path).resolve()
    try:
        ep.relative_to(REPO_ROOT)
    except ValueError:
        finalize_job(
            REPO_ROOT,
            job_id,
            kind="cert",
            request=req,
            exit_code=1,
            error="episode path outside repo root",
        )
        return
    if not ep.is_file():
        finalize_job(
            REPO_ROOT,
            job_id,
            kind="cert",
            request=req,
            exit_code=1,
            error="episode file not found",
        )
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
    try:
        r = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=7200,
        )
        finalize_job(
            REPO_ROOT,
            job_id,
            kind="cert",
            request=req,
            exit_code=r.returncode,
            stdout=r.stdout or "",
            stderr=r.stderr or "",
        )
    except subprocess.TimeoutExpired as e:
        finalize_job(
            REPO_ROOT,
            job_id,
            kind="cert",
            request=req,
            exit_code=-1,
            stderr=str(e),
            error="subprocess timeout (7200s)",
        )
    except Exception as e:
        finalize_job(
            REPO_ROOT,
            job_id,
            kind="cert",
            request=req,
            exit_code=-1,
            error=str(e)[:4000],
        )


# 須在 GET /api/runs/{run_id} 之前註冊，否則部分環境下 /api/runs/wazuh 會被當成 {run_id} 而 POST 得到 405
@app.post("/api/runs/wazuh")
def api_run_wazuh(request: Request, body: WazuhRunBody, background_tasks: BackgroundTasks):
    """背景啟動 Wazuh Indexer → Episode → Layer C（需本機已設定 WAZUH_* 與 DATABASE_URL 等，與 CLI 相同）。"""
    _check_api_key(request)
    job_id = uuid.uuid4().hex
    write_job_running(REPO_ROOT, job_id, "wazuh", _wazuh_request_dict(body))
    background_tasks.add_task(_background_wazuh, body, job_id)
    return {
        "ok": True,
        "message": "wazuh pipeline started in background",
        "job_id": job_id,
        "target_ip": body.target_ip,
        "hours": body.hours,
        "evidenceops": body.evidenceops,
    }


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
    job_id = uuid.uuid4().hex
    write_job_running(REPO_ROOT, job_id, "cert", _cert_request_dict(body))
    background_tasks.add_task(_background_cert, body, job_id)
    return {
        "ok": True,
        "message": "cert pipeline started in background",
        "job_id": job_id,
        "episode_path": body.episode_path,
    }


@app.get("/api/pipeline-jobs/{job_id}")
def api_pipeline_job(job_id: str):
    """輪詢背景管線狀態：running | ok | failed。"""
    if not validate_job_id(job_id):
        raise HTTPException(status_code=400, detail="invalid job_id")
    data = read_job(REPO_ROOT, job_id)
    if not data:
        raise HTTPException(status_code=404, detail="job not found")
    return data


def _investigation_spa_index() -> FileResponse | HTMLResponse:
    """Layer C Investigation Graph（Vite SPA；需先 build mvp-investigation-spa）。"""
    index = INVESTIGATION_STATIC / "index.html"
    if not index.is_file():
        return HTMLResponse(
            "<html><body><h1>Investigation UI not built</h1>"
            "<p>Run from repo root:</p><pre>"
            "cd packages/layerc-graph-ui && npm ci && npm run build\n"
            "cd packages/mvp-investigation-spa && npm ci && npm run build"
            "</pre></body></html>",
            status_code=503,
        )
    return FileResponse(index)


@app.get("/investigations/graph", response_class=HTMLResponse)
def investigation_graph_spa():
    return _investigation_spa_index()


@app.get("/investigations/graph/{rest:path}", response_class=HTMLResponse)
def investigation_graph_spa_deeplink(rest: str):
    """與根路徑相同 SPA；支援路徑帶 episode_id 的深連結。"""
    _ = rest
    return _investigation_spa_index()


@app.get("/api/investigations/cases")
def api_investigation_cases():
    from .investigation import run_row_to_episode_list_entry

    rows = list_runs(_db_url(), limit=200, offset=0)
    return [run_row_to_episode_list_entry(dict(r)) for r in rows]


@app.get("/api/investigations/case/by-episode/{episode_id}")
def api_investigation_case_by_episode(episode_id: str):
    from .investigation import run_row_to_layerc_payload

    if not episode_id or ".." in episode_id or "/" in episode_id or "\\" in episode_id:
        raise HTTPException(status_code=400, detail="invalid episode_id")
    row = get_run_by_episode_id(_db_url(), episode_id)
    if not row:
        raise HTTPException(status_code=404)
    return run_row_to_layerc_payload(row)


@app.get("/api/investigations/case/{run_id}")
def api_investigation_case(run_id: int):
    from .investigation import run_row_to_layerc_payload

    row = get_run_by_id(_db_url(), run_id)
    if not row:
        raise HTTPException(status_code=404)
    return run_row_to_layerc_payload(row)


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


def _safe_evidenceops_episode_id(episode_id: str) -> None:
    if not episode_id or ".." in episode_id or "/" in episode_id or "\\" in episode_id:
        raise HTTPException(status_code=400, detail="Invalid episode_id")


@app.get("/api/evidenceops/{episode_id}/{kind}")
def api_evidenceops_artifact(episode_id: str, kind: str):
    """EvidenceOps 產物：decision_bundle、case_summary（相對 outputs/evidenceops/）。"""
    _safe_evidenceops_episode_id(episode_id)
    if kind not in ("decision_bundle", "case_summary"):
        raise HTTPException(status_code=400, detail="kind must be decision_bundle|case_summary")
    base = REPO_ROOT / "outputs" / "evidenceops"
    name = f"decision_bundle_{episode_id}.json" if kind == "decision_bundle" else f"case_summary_{episode_id}.json"
    path = (base / name).resolve()
    try:
        path.relative_to((REPO_ROOT / "outputs").resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid path") from None
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path, media_type="application/json", filename=path.name)


if INVESTIGATION_STATIC.is_dir():
    app.mount(
        "/static/investigation",
        StaticFiles(directory=str(INVESTIGATION_STATIC)),
        name="investigation_static",
    )
