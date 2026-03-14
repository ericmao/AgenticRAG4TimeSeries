#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EPISODE_PATH="tests/demo/episode_insider_highrisk.json"

echo "=== High-Risk Insider Demo ==="

# a) Generate episode
python3 scripts/gen_insider_highrisk_episode.py \
  --out_path "$EPISODE_PATH" \
  --user USER0420 \
  --hosts PC010,PC011,PC012,PC013 \
  --t0_ms 1771754000000 \
  --window_ms 3600000 \
  --burst_events 60 \
  --device_churn 12

# b) Retrieve evidence
python3 -m src.cli retrieve --episode "$EPISODE_PATH"

# c) Analyze (triage, hunt, response)
python3 -m src.cli analyze --episode "$EPISODE_PATH"

# d) Writeback
python3 -m src.cli writeback --episode "$EPISODE_PATH"

# e) Eval (episodes_dir = tests/demo, limit 10)
python3 -m src.cli eval --episodes_dir tests/demo --limit 10

# f) Demo report
python3 -m src.cli demo_report --episode "$EPISODE_PATH"

EPISODE_ID="cert-USER0420-highrisk"
echo ""
echo "=== Artifact paths ==="
echo "  evidence:    outputs/evidence/${EPISODE_ID}.json"
echo "  agents:      outputs/agents/${EPISODE_ID}_triage.json"
echo "               outputs/agents/${EPISODE_ID}_hunt_planner.json"
echo "               outputs/agents/${EPISODE_ID}_response_advisor.json"
echo "  writeback:   outputs/writeback/${EPISODE_ID}.json"
echo "  eval:        outputs/eval/metrics.csv"
echo "  demo report: outputs/demo/demo_report.md"
