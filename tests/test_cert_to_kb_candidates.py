"""
Tests for CERT/Episode → KB candidates pipeline and self-evaluation (reliability_score, auto_add).
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.contracts.episode import Episode
from src.contracts.kb_candidate import KBCandidate
from src.pipeline.cert_to_kb_candidates import (
    compute_reliability_score,
    export_candidates_to_kb,
    generate_candidate,
    run_cert_to_kb_candidates,
)


def test_compute_reliability_score_high_signal():
    """Lateral + burst + multiple hosts yields higher score."""
    score = compute_reliability_score(
        sequence_tags=["lateral", "burst", "device"],
        event_count=80,
        artifacts_summary={"user": ["U1"], "host": ["PC-1", "PC-2", "PC-3"]},
    )
    assert score >= 0.5
    assert score <= 1.0


def test_compute_reliability_score_low_signal():
    """Only logon, few events, one host yields lower score."""
    score = compute_reliability_score(
        sequence_tags=["logon"],
        event_count=5,
        artifacts_summary={"user": ["U1"], "host": ["PC-1"]},
    )
    assert score < 0.5
    assert score >= 0.0


def test_generate_candidate_sets_reliability_and_auto_add():
    """generate_candidate sets reliability_score and auto_add from threshold."""
    ep = Episode(
        episode_id="cert-U1-w0",
        run_id="r1",
        t0_ms=0,
        t1_ms=1,
        entities=["U1", "PC-1", "PC-2"],
        artifacts=[{"type": "user", "value": "U1"}, {"type": "host", "value": "PC-1"}, {"type": "host", "value": "PC-2"}],
        sequence_tags=["lateral", "logon"],
        events=[{"ts_ms": i} for i in range(20)],
    )
    c_low = generate_candidate(ep, reliability_threshold=0.9)
    c_high = generate_candidate(ep, reliability_threshold=0.2)
    assert 0 <= c_low.reliability_score <= 1
    assert c_low.episode_id == "cert-U1-w0"
    assert c_low.body_md.startswith("## Episode:")
    assert "lateral" in c_low.sequence_tags
    assert c_high.auto_add is True
    assert c_low.reliability_score == c_high.reliability_score


def test_run_cert_to_kb_candidates_and_export():
    """Run pipeline on fixture episodes; export only score >= threshold to kb."""
    with tempfile.TemporaryDirectory() as tmp:
        ep_dir = Path(tmp) / "episodes"
        ep_dir.mkdir()
        for i, (tags, n_events) in enumerate([(["lateral", "burst"], 50), (["logon"], 3)]):
            ep = Episode(
                episode_id=f"cert-U-w{i}",
                run_id="r1",
                t0_ms=0,
                t1_ms=1,
                entities=["U"],
                artifacts=[{"type": "user", "value": "U"}, {"type": "host", "value": "PC-1"}],
                sequence_tags=tags,
                events=[{"ts_ms": j} for j in range(n_events)],
            )
            (ep_dir / f"cert-U-w{i}.json").write_text(ep.model_dump_json(indent=2), encoding="utf-8")
        out_dir = Path(tmp) / "out"
        candidates, out_path = run_cert_to_kb_candidates(
            episodes_dir=ep_dir,
            out_dir=out_dir,
            limit=10,
            reliability_threshold=0.4,
        )
        assert len(candidates) == 2
        assert out_path.exists()
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert len(data) == 2
        assert all("reliability_score" in d and "auto_add" in d for d in data)
        kb_dir_path, written = export_candidates_to_kb(
            out_path, kb_subdir="cert_candidates_test", reliability_threshold=0.4, max_export=10
        )
        assert written >= 1
        assert kb_dir_path.exists()
        md_files = list(kb_dir_path.glob("*.md"))
        assert len(md_files) >= 1
