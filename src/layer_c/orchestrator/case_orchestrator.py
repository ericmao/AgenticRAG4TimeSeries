"""
Case Orchestrator — SenseL EvidenceOps multi-agent Layer C flow.
Evidence-grounded only; delegates triage/hunt/response to existing agents via run_single_agent.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from src.contracts.agent_output import AgentOutput
from src.contracts.episode import Episode
from src.contracts.evidence import EvidenceSet
from src.layer_c.agents.cti_correlation_agent import run_cti_correlation
from src.layer_c.agents.entity_investigation_agent import run_entity_investigation
from src.layer_c.orchestrator import routing_policy
from src.layer_c.orchestrator.output_writer import write_agent_bundle_to_disk
from src.layer_c.schemas.decision_bundle import CaseState, HypothesisInput, RiskContextInput
from src.layer_c.telemetry import finalize_run, record_agent_done, record_agent_start, reset_run
from src.layer_c.validators.decision_validator import validate_case_state
from src.pipeline.run_agents import run_single_agent

ORCHESTRATOR_VERSION = "0.1.0"


def _stub_hunt_planner(episode: Episode, evidence_set: EvidenceSet, reason: str) -> AgentOutput:
    items = evidence_set.items
    citation_ids: list[str] = []
    for i in range(min(5, len(items))):
        citation_ids.append(items[i].evidence_id)
    while len(citation_ids) < 3 and len(citation_ids) < len(items):
        citation_ids.append(items[len(citation_ids)].evidence_id)
    return AgentOutput(
        agent_id="hunt_planner",
        episode_id=episode.episode_id,
        run_id=episode.run_id,
        summary=f"Hunt planner skipped ({reason}); no new queries generated.",
        confidence=0.5,
        citations=citation_ids[: max(3, len(citation_ids))] if citation_ids else [],
        assumptions=[],
        next_required_data=[],
        structured={"skipped": True, "reason": reason, "queries": [], "pivots": [], "expected_findings": []},
    )


def _merge_trust(
    episode: Episode,
    risk: Optional[RiskContextInput],
    hypothesis: Optional[HypothesisInput],
) -> dict[str, Any]:
    ts: dict[str, Any] = {}
    if risk is not None:
        ts.update(risk.raw)
    else:
        ts.update(RiskContextInput.from_episode(episode).raw)
    if hypothesis is not None:
        ts.update(hypothesis.to_trust_signals())
    return ts


class CaseOrchestrator:
    def __init__(self, repo_root: Optional[Path] = None):
        self.repo_root = repo_root or Path(__file__).resolve().parents[2]

    def run(
        self,
        episode: Episode,
        evidence_set: EvidenceSet,
        risk: Optional[RiskContextInput] = None,
        hypothesis: Optional[HypothesisInput] = None,
        rule_id: str = "default",
        *,
        write_outputs: bool = True,
    ) -> tuple[CaseState, str]:
        """
        Execute triage → (entity, cti) → hunt → response per routing_policy.
        Returns (CaseState, status) where status is \"ok\" or \"failed\".
        """
        trust = _merge_trust(episode, risk, hypothesis)
        bundle: dict[str, AgentOutput] = {}
        root = self.repo_root

        reset_run(root, episode.episode_id, episode.run_id)

        def _run_agent(name: str, fn: Any) -> AgentOutput:
            record_agent_start(root, name, f"executing {name}")
            out = fn()
            record_agent_done(root, name, detail=f"completed {name}")
            return out

        try:
            bundle["triage"] = _run_agent(
                "triage",
                lambda: run_single_agent(
                    "triage",
                    episode,
                    evidence_set,
                    trust_signals=trust,
                    repair_hint=None,
                    write_outputs=False,
                    rule_id=rule_id,
                ),
            )

            triage_out = bundle["triage"]
            structured = triage_out.structured if isinstance(triage_out.structured, dict) else {}
            tl = routing_policy.triage_level_from_output(structured)
            route = routing_policy.plan_route(tl)

            bundle["entity_investigation"] = _run_agent(
                "entity_investigation",
                lambda: run_entity_investigation(
                    episode, evidence_set, trust_signals=trust, repair_hint=None
                ),
            )

            if route.get("run_cti_correlation"):
                bundle["cti_correlation"] = _run_agent(
                    "cti_correlation",
                    lambda: run_cti_correlation(
                        episode, evidence_set, trust_signals=trust, repair_hint=None
                    ),
                )
            else:
                record_agent_start(root, "cti_correlation", "skipped by routing")
                record_agent_done(
                    root, "cti_correlation", detail="not run (noise tier)", skipped=True
                )

            if route.get("run_hunt_planner"):
                bundle["hunt_planner"] = _run_agent(
                    "hunt_planner",
                    lambda: run_single_agent(
                        "hunt_planner",
                        episode,
                        evidence_set,
                        trust_signals=trust,
                        repair_hint=None,
                        write_outputs=False,
                        rule_id=rule_id,
                    ),
                )
            else:
                record_agent_start(root, "hunt_planner", "stub (routing)")
                hp = _stub_hunt_planner(
                    episode,
                    evidence_set,
                    str(route.get("hunt_stub_reason") or "routing"),
                )
                bundle["hunt_planner"] = hp
                record_agent_done(root, "hunt_planner", detail="stub hunt_planner")

            if route.get("run_response_advisor"):
                bundle["response_advisor"] = _run_agent(
                    "response_advisor",
                    lambda: run_single_agent(
                        "response_advisor",
                        episode,
                        evidence_set,
                        trust_signals=trust,
                        repair_hint=None,
                        write_outputs=False,
                        rule_id=rule_id,
                    ),
                )

            state = CaseState(
                episode_id=episode.episode_id,
                run_id=episode.run_id,
                rule_id=rule_id,
                orchestrator_version=ORCHESTRATOR_VERSION,
                routing=route,
                by_agent_id=bundle,
            )

            ok, errors = validate_case_state(state, episode, evidence_set)
            if not ok:
                state.routing["validation_errors"] = errors
                finalize_run(root, False, "; ".join(errors[:5]))
                return state, "failed"

            finalize_run(root, True, "validation passed")
            if write_outputs:
                write_agent_bundle_to_disk(episode.episode_id, rule_id, bundle, self.repo_root)

            return state, "ok"
        except Exception as e:
            finalize_run(root, False, str(e)[:500])
            raise
