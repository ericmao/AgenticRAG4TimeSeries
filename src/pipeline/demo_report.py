"""
Generate a full end-to-end demo report at outputs/demo/demo_report.md.
Loads episode, evidence, agent outputs, writeback, eval; warns on missing artifacts.
"""
from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPORT_PATH = Path("outputs/demo/demo_report.md")

# Redact patterns: sk- keys, OpenCTI token-like (long hex), emails (optional)
REDACT_SK = re.compile(r"sk-[a-zA-Z0-9]{8,}", re.IGNORECASE)
REDACT_TOKEN = re.compile(r"\b[a-f0-9]{32,64}\b")
REDACT_EMAIL = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")


def truncate(s: str, max_len: int = 200) -> str:
    """Safely truncate long strings."""
    if not s:
        return ""
    s = str(s).strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3].rstrip() + "..."


def redact(s: str, redact_emails: bool = False) -> str:
    """Redact obvious secrets; keep artifacts as-is unless explicitly [REDACTED]."""
    if not s or "[REDACTED]" in s:
        return s
    out = REDACT_SK.sub("[REDACTED]", s)
    # Only redact long hex if it looks like a token (e.g. 32+ hex), not IPs/hashes we want to show
    # Be conservative: redact only sk- and leave hex hashes (evidence_id) as-is for report
    if redact_emails:
        out = REDACT_EMAIL.sub("[REDACTED]", out)
    return out


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def generate_demo_report(episode_path: str | Path) -> Path:
    """
    Load episode and optional artifacts from outputs/; generate outputs/demo/demo_report.md.
    Returns path to the report file.
    """
    root = _repo_root()
    episode_path = Path(episode_path)
    if not episode_path.is_absolute():
        episode_path = root / episode_path
    if not episode_path.exists():
        raise FileNotFoundError(f"Episode file not found: {episode_path}")

    episode_data = _load_json(episode_path)
    if not episode_data:
        raise ValueError(f"Invalid episode JSON: {episode_path}")
    episode_id = episode_data.get("episode_id", "unknown")
    run_id = episode_data.get("run_id", "")

    out_dir = root / "outputs" / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "demo_report.md"
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    warnings: list[str] = []
    evidence = _load_json(root / "outputs" / "evidence" / f"{episode_id}.json")
    if not evidence:
        warnings.append("Evidence not found (run retrieve first).")
    triage = _load_json(root / "outputs" / "agents" / f"{episode_id}_triage.json")
    hunt = _load_json(root / "outputs" / "agents" / f"{episode_id}_hunt_planner.json")
    response = _load_json(root / "outputs" / "agents" / f"{episode_id}_response_advisor.json")
    if not triage:
        warnings.append("Triage agent output not found (run analyze first).")
    if not hunt:
        warnings.append("Hunt planner agent output not found.")
    if not response:
        warnings.append("Response advisor agent output not found.")
    writeback = _load_json(root / "outputs" / "writeback" / f"{episode_id}.json")
    if not writeback:
        warnings.append("Writeback patch not found (run writeback first).")

    metrics_row: Optional[dict[str, str]] = None
    metrics_path = root / "outputs" / "eval" / "metrics.csv"
    if metrics_path.exists():
        with open(metrics_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("episode_id") == episode_id:
                    metrics_row = row
                    break

    # Build markdown
    lines: list[str] = []
    lines.append(f"# Demo Report: {episode_id}")
    lines.append("")
    lines.append(f"- **run_id**: {run_id}")
    lines.append(f"- **generated_at**: {generated_at}")
    lines.append("")

    if warnings:
        lines.append("## Warnings")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    lines.append("## Episode summary")
    lines.append("")
    entities = episode_data.get("entities") or []
    artifacts = episode_data.get("artifacts") or []
    tags = episode_data.get("sequence_tags") or []
    t0 = episode_data.get("t0_ms")
    t1 = episode_data.get("t1_ms")
    lines.append(f"- **entities**: {entities}")
    lines.append(f"- **artifacts**: {len(artifacts)} items")
    for a in artifacts[:10]:
        lines.append(f"  - {a.get('type', '?')}: `{redact(str(a.get('value', '')))}`")
    if len(artifacts) > 10:
        lines.append(f"  - ... and {len(artifacts) - 10} more")
    lines.append(f"- **sequence_tags**: {tags}")
    if t0 is not None and t1 is not None:
        lines.append(f"- **time window**: {t0} .. {t1} (ms)")
    lines.append("")

    lines.append("## Evidence summary")
    lines.append("")
    if evidence:
        items = evidence.get("items") or []
        stats = evidence.get("stats") or {}
        lines.append(f"- **items_count**: {len(items)}")
        lines.append(f"- **by_source**: {stats.get('by_source', {})}")
        lines.append(f"- **by_kind**: {stats.get('by_kind', {})}")
        lines.append("")
        lines.append("Top 5 evidence items:")
        for item in items[:5]:
            eid = item.get("evidence_id", "")[:12] + "..."
            title = truncate(item.get("title", ""), 60)
            excerpt = truncate(item.get("body", ""), 80).replace("\n", " ")
            lines.append(f"- `{eid}` **{title}** — {redact(excerpt)}")
    else:
        lines.append("*(No evidence loaded)*")
    lines.append("")

    lines.append("## Agent outputs")
    lines.append("")
    if triage:
        struct = triage.get("structured") or {}
        lines.append("### Triage")
        lines.append(f"- **triage_level**: {struct.get('triage_level', '—')}")
        lines.append(f"- **confidence**: {triage.get('confidence', '—')}")
        cites = triage.get("citations") or []
        lines.append(f"- **citations** (first 5): {cites[:5]}")
        lines.append("")
    if hunt:
        struct = hunt.get("structured") or {}
        q = struct.get("queries") or {}
        lines.append("### Hunt planner")
        by_platform = {k: len(v) if isinstance(v, list) else 0 for k, v in q.items()}
        lines.append(f"- **queries by platform**: {by_platform}")
        example_queries: list[str] = []
        for v in q.values() if isinstance(q, dict) else []:
            if isinstance(v, list):
                for x in v[:2]:
                    if isinstance(x, str):
                        example_queries.append(truncate(redact(x), 100))
                    elif isinstance(x, dict):
                        example_queries.append(truncate(redact(json.dumps(x)), 100))
        lines.append("- **example queries**:")
        for eq in example_queries[:3]:
            lines.append(f"  - `{eq}`")
        lines.append("")
    if response:
        struct = response.get("structured") or {}
        actions = struct.get("actions") or []
        lines.append("### Response advisor")
        lines.append("- **actions**:")
        for ac in actions[:10]:
            if isinstance(ac, dict):
                a = ac.get("action", "—")
                t = truncate(redact(str(ac.get("target", ""))), 50)
                d = ac.get("duration_minutes")
                lines.append(f"  - {a} | target: {t} | duration_minutes: {d}")
        if actions and isinstance(actions[0], dict):
            lines.append(f"- **guardrails** (first action): {truncate(str(actions[0].get('guardrails', '')), 120)}")
        cites = response.get("citations") or []
        lines.append(f"- **citations** (first 5): {cites[:5]}")
    lines.append("")

    lines.append("## Validation status")
    lines.append("")
    lines.append("Outputs were validated by pipeline (citations + policy guardrails); see `analyze` and `validate_and_repair`.")
    lines.append("")

    lines.append("## Writeback summary")
    lines.append("")
    if writeback:
        lines.append(f"- **mode**: {writeback.get('mode', '—')}")
        stats = writeback.get("stats") or {}
        lines.append(f"- **stats**: {json.dumps(stats)}")
        notes = writeback.get("notes") or []
        if notes:
            n0 = notes[0] if isinstance(notes[0], dict) else {}
            lines.append(f"- **first note**: type={n0.get('type', '—')}, agent_id={n0.get('agent_id', '—')}, citations_count={len(n0.get('citations') or [])}")
    else:
        lines.append("*(No writeback loaded)*")
    lines.append("")

    lines.append("## Eval snapshot")
    lines.append("")
    if metrics_row:
        lines.append(f"- **pass**: {metrics_row.get('pass', '—')}")
        lines.append(f"- **evidence_items**: {metrics_row.get('evidence_items', '—')}")
        lines.append(f"- **egr_overall**: {metrics_row.get('egr_overall', '—')}")
        lines.append(f"- **ucr_proxy**: {metrics_row.get('ucr_proxy', '—')}")
        lines.append(f"- **total_ms**: {metrics_row.get('total_ms', '—')}")
    else:
        lines.append("*(No row in metrics.csv for this episode_id; run eval first.)*")
    lines.append("")

    lines.append("## How to reproduce")
    lines.append("")
    try:
        rel_ep = episode_path.relative_to(root)
    except ValueError:
        rel_ep = Path(episode_path.name)
    lines.append("```bash")
    lines.append(f"python -m src.cli retrieve --episode {rel_ep}")
    lines.append(f"python -m src.cli analyze --episode {rel_ep}")
    lines.append(f"python -m src.cli writeback --episode {rel_ep}")
    lines.append("python -m src.cli eval --episodes_dir tests/samples/episodes --limit 20")
    lines.append(f"python -m src.cli demo_report --episode {rel_ep}")
    lines.append("```")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
