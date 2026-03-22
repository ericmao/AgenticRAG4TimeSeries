#!/usr/bin/env bash
# SenseL Layer C — single episode demo: retrieve → analyze → writeback [→ POST /api/v1/cases/writeback]
# Usage: bash scripts/run_layerc_episode_demo.sh [path/to/episode.json]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EPISODE_PATH="${1:-tests/demo/episode_insider_highrisk.json}"

echo "=== Layer C Episode Demo ==="
echo "  episode: $EPISODE_PATH"

# 1) Retrieve evidence
python3 -m src.cli retrieve --episode "$EPISODE_PATH"

# 2) Analyze (triage, hunt, response)
python3 -m src.cli analyze --episode "$EPISODE_PATH"

# 3) Writeback (dry-run decision bundle)
python3 -m src.cli writeback --episode "$EPISODE_PATH"

# 4) Optional: eval + demo_report (if episode is under an episodes_dir)
# python3 -m src.cli eval --episodes_dir "$(dirname "$EPISODE_PATH")" --limit 1
# python3 -m src.cli demo_report --episode "$EPISODE_PATH"

# 5) Optional — SenseL EvidenceOps multi-agent orchestrator (entity/CTI + same writeback path):
#    PYTHONPATH=. python3 scripts/run_layerc_case_orchestrator.py --episode "$EPISODE_PATH"
#    See docs/LAYER_C_EVIDENCEOPS.md

echo "Done. Outputs: outputs/evidence/, outputs/agents/, outputs/writeback/, outputs/audit/"
echo "To POST decision bundle to Control Plane: implement client in connectors and call POST /api/v1/cases/writeback"
