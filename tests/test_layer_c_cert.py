"""
Layer C pipeline tests using CERT-style data (r3.2 shape: logon/device with pc column).
Builds episodes from minimal r3.2-style CSVs, then runs retrieve -> analyze (agents) and asserts outputs.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_r32_style_cert_fixture(tmp_path: Path) -> Path:
    """Write minimal logon.csv + device.csv with 'pc' column (r3.2 style). Returns data_dir."""
    (tmp_path / "logon.csv").write_text(
        "id,date,user,pc,activity\n"
        "id1,01/01/2010 06:20:00,USER1,PC-001,Logon\n"
        "id2,01/01/2010 06:30:00,USER1,PC-001,Logoff\n"
        "id3,01/01/2010 07:00:00,USER1,PC-002,Logon\n"
        "id4,01/02/2010 08:00:00,USER1,PC-001,Logon\n"
        "id5,01/02/2010 09:00:00,USER1,PC-001,Logoff\n"
        "id6,01/08/2010 10:00:00,USER1,PC-001,Logon\n",
        encoding="utf-8",
    )
    (tmp_path / "device.csv").write_text(
        "id,date,user,pc,activity\n"
        "d1,01/01/2010 06:45:00,USER1,PC-001,Connect\n"
        "d2,01/01/2010 07:15:00,USER1,PC-002,Disconnect\n",
        encoding="utf-8",
    )
    return tmp_path


def test_layer_c_retrieve_and_analyze_on_cert_episode():
    """Build one episode from r3.2-style CERT data, run retrieve + agents; assert evidence and citations."""
    from src.pipeline.cert_to_episodes import build_episodes_from_cert
    from src.pipeline.retrieve_evidence import build_evidence_set
    from src.pipeline.run_agents import run_all_agents
    from src.contracts.episode import Episode
    from src.contracts.evidence import EvidenceSet

    with tempfile.TemporaryDirectory() as tmp:
        data_dir = _make_r32_style_cert_fixture(Path(tmp))
        episodes = build_episodes_from_cert(
            data_dir, run_id="cert-r32-test", window_days=7, burst_threshold=50
        )
        assert len(episodes) >= 1, "need at least one episode from fixture"
        ep = episodes[0]
        episode_path = Path(tmp) / f"{ep.episode_id}.json"
        episode_path.write_text(ep.model_dump_json(indent=2), encoding="utf-8")

        # Retrieve: writes to REPO_ROOT/outputs/evidence/<episode_id>.json
        build_evidence_set(episode_path)
        evidence_path = REPO_ROOT / "outputs" / "evidence" / f"{ep.episode_id}.json"
        assert evidence_path.exists(), "evidence file should be written"
        evidence_set = EvidenceSet.model_validate(
            json.loads(evidence_path.read_text(encoding="utf-8"))
        )
        assert len(evidence_set.items) >= 1, "should retrieve at least one evidence item"

        # Analyze: run agents (no LLM, write_outputs=False to keep test hermetic)
        outputs = run_all_agents(
            ep, evidence_set, trust_signals=None, repair_hint=None, write_outputs=False
        )
        assert "triage" in outputs
        assert "hunt_planner" in outputs
        assert "response_advisor" in outputs
        for agent_id, out in outputs.items():
            assert len(out.citations) >= 3, f"{agent_id} should have at least 3 citations"


def test_layer_c_eval_one_cert_episode():
    """Run full _run_one_episode path (retrieve + agents + validate + writeback) on one CERT-style episode."""
    from src.pipeline.cert_to_episodes import build_episodes_from_cert
    from src.eval.run_eval import _run_one_episode, _repo_root

    with tempfile.TemporaryDirectory() as tmp:
        data_dir = _make_r32_style_cert_fixture(Path(tmp))
        episodes = build_episodes_from_cert(
            data_dir, run_id="cert-r32-eval", window_days=7, burst_threshold=50
        )
        assert len(episodes) >= 1
        ep = episodes[0]
        episode_path = Path(tmp) / f"{ep.episode_id}.json"
        episode_path.write_text(ep.model_dump_json(indent=2), encoding="utf-8")
        root = _repo_root()
        row, issues = _run_one_episode(episode_path, root)
        assert row is not None
        assert row.get("episode_id") == ep.episode_id
        assert row.get("evidence_items", 0) >= 1
        assert row.get("pass") == 1, f"episode should pass validation: {issues}"
