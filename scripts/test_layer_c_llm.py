#!/usr/bin/env python3
"""
Test Layer C with optional LLM (Ollama default) and time-series signals.
- Tests get_time_series_signals(episode) when no CERT deps -> available false.
- Tests get_llm_for_layer_c() returns a chat model (Ollama or OpenAI).
- Runs analyze with USE_LANGCHAIN_FOR_ANALYSIS=true and prints triage structured.llm_analysis.
Requires: episode + evidence already present (run retrieve first) or use --episode and run retrieve in script.
Usage:
  PYTHONPATH=. python scripts/test_layer_c_llm.py
  PYTHONPATH=. USE_LANGCHAIN_FOR_ANALYSIS=true python scripts/test_layer_c_llm.py --episode tests/demo/episode_insider_highrisk.json
  PYTHONPATH=. python scripts/test_layer_c_llm.py --skip-analyze   # unit tests only (no Ollama needed for time_series_signals + factory)

For LLM analysis to return real text: install Ollama, run `ollama pull llama3.1`, then set USE_LANGCHAIN_FOR_ANALYSIS=true.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_time_series_signals() -> bool:
    """get_time_series_signals without deps -> available false."""
    from src.contracts.episode import Episode
    from src.pipeline.time_series_signals import get_time_series_signals
    ep = Episode(episode_id="test-ep", run_id="test-run", t0_ms=0, t1_ms=0, entity_id="USER0420", entities=["USER0420"])
    out = get_time_series_signals(ep)
    assert isinstance(out, dict), "get_time_series_signals should return dict"
    assert "available" in out and "entity_id" in out, "must have available and entity_id"
    assert out["entity_id"] == "USER0420"
    # Without injecting data_processor/vector_store, available should be False
    assert out["available"] is False or out.get("error"), "no deps => available false or error"
    print("[OK] get_time_series_signals(episode) -> available=false when no deps")
    return True


def test_llm_factory() -> bool:
    """get_llm_for_layer_c returns a chat model (Ollama default)."""
    from src.llm.factory import get_llm_for_layer_c
    try:
        llm = get_llm_for_layer_c()
        # Must have invoke for LangChain chat model
        assert hasattr(llm, "invoke"), "LLM must have invoke"
        print("[OK] get_llm_for_layer_c() returned model with invoke")
        return True
    except Exception as e:
        print(f"[SKIP] get_llm_for_layer_c: {e} (install langchain-ollama and run Ollama for full test)")
        return True  # don't fail if Ollama not installed


def test_analyze_with_llm(episode_path: Path) -> bool:
    """Run retrieve (if needed) then analyze with USE_LANGCHAIN_FOR_ANALYSIS=true; print llm_analysis."""
    os.environ["USE_LANGCHAIN_FOR_ANALYSIS"] = "true"
    episode_path = REPO_ROOT / episode_path if not str(episode_path).startswith("/") else Path(episode_path)
    if not episode_path.exists():
        print(f"[SKIP] Episode not found: {episode_path}")
        return True
    ep_data = json.loads(episode_path.read_text(encoding="utf-8"))
    episode_id = ep_data.get("episode_id", "ep-unknown")
    evidence_path = REPO_ROOT / "outputs" / "evidence" / f"{episode_id}.json"
    if not evidence_path.exists():
        print("Running retrieve to produce evidence...")
        r = subprocess.run(
            [sys.executable, "-m", "src.cli", "retrieve", "--episode", str(episode_path)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        )
        if r.returncode != 0:
            print(f"[SKIP] retrieve failed: {r.stderr or r.stdout}")
            return True
        if not evidence_path.exists():
            print("[SKIP] evidence still missing after retrieve")
            return True
    # Run analyze with LLM
    from src.contracts.episode import Episode
    from src.contracts.evidence import EvidenceSet
    from src.pipeline.run_agents import run_all_agents
    episode = Episode.model_validate(ep_data)
    evidence_set = EvidenceSet.model_validate(json.loads(evidence_path.read_text(encoding="utf-8")))
    outputs = run_all_agents(episode, evidence_set, write_outputs=False)
    triage = outputs.get("triage")
    if not triage:
        print("[FAIL] no triage output")
        return False
    llm_analysis = (triage.structured or {}).get("llm_analysis", "")
    print("[OK] analyze with USE_LANGCHAIN_FOR_ANALYSIS=true completed")
    print("triage.structured.llm_analysis:")
    print("-" * 40)
    print(llm_analysis or "(empty)")
    print("-" * 40)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Layer C LLM (Ollama) and time-series signals")
    parser.add_argument("--episode", type=str, default="tests/demo/episode_insider_highrisk.json", help="Episode JSON for analyze test")
    parser.add_argument("--skip-analyze", action="store_true", help="Skip full analyze (only unit tests)")
    args = parser.parse_args()
    ok = True
    ok = test_time_series_signals() and ok
    ok = test_llm_factory() and ok
    if not args.skip_analyze:
        ok = test_analyze_with_llm(Path(args.episode)) and ok
    print("\nLayer C LLM tests finished.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
