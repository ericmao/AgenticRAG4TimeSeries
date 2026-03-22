"""EvidenceOps stub agents produce valid citations when evidence exists."""
from __future__ import annotations

import json
from pathlib import Path

from src.contracts.episode import Episode
from src.contracts.evidence import EvidenceSet
from src.layer_c.agents.cti_correlation_agent import run_cti_correlation
from src.layer_c.agents.entity_investigation_agent import run_entity_investigation

_REPO = Path(__file__).resolve().parents[1]


def test_entity_and_cti_min_citations():
    ep_path = _REPO / "tests" / "demo" / "episode_insider_highrisk.json"
    ev_path = _REPO / "outputs" / "evidence" / "cert-USER0420-highrisk.json"
    if not ev_path.exists():
        return
    episode = Episode.model_validate(json.loads(ep_path.read_text()))
    evidence_set = EvidenceSet.model_validate(json.loads(ev_path.read_text()))
    e = run_entity_investigation(episode, evidence_set)
    c = run_cti_correlation(episode, evidence_set)
    assert len(e.citations) >= 3
    assert len(c.citations) >= 3
    assert e.agent_id == "entity_investigation"
    assert c.agent_id == "cti_correlation"
