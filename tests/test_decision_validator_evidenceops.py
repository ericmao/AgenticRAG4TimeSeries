"""Decision validator on synthetic CaseState."""
from __future__ import annotations

import json
from pathlib import Path

from src.contracts.episode import Episode
from src.contracts.evidence import EvidenceSet
from src.layer_c.orchestrator.case_orchestrator import CaseOrchestrator
from src.layer_c.schemas.decision_bundle import RiskContextInput
from src.layer_c.validators.decision_validator import validate_case_state

_REPO = Path(__file__).resolve().parents[1]


def test_validate_case_state_after_orchestrator():
    ep_path = _REPO / "tests" / "demo" / "episode_insider_highrisk.json"
    ev_path = _REPO / "outputs" / "evidence" / "cert-USER0420-highrisk.json"
    if not ev_path.exists():
        return
    episode = Episode.model_validate(json.loads(ep_path.read_text()))
    evidence_set = EvidenceSet.model_validate(json.loads(ev_path.read_text()))
    orch = CaseOrchestrator(repo_root=_REPO)
    state, status = orch.run(
        episode,
        evidence_set,
        risk=RiskContextInput.from_episode(episode),
        write_outputs=False,
    )
    assert status == "ok"
    ok, errs = validate_case_state(state, episode, evidence_set)
    assert ok is True
    assert errs == []
