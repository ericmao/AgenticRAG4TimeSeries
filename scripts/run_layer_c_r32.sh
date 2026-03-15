#!/usr/bin/env bash
# Run Layer C pipeline using CERT r3.2 data:
#   1) cert2episodes from r3.2 -> outputs/episodes/cert_r32
#   2) eval on those episodes (retrieve -> analyze -> writeback -> metrics) with --limit
# Usage:
#   R32_DATA_DIR=/path/to/r3.2 bash scripts/run_layer_c_r32.sh
#   R32_DATA_DIR=/path/to/r3.2 LIMIT=5 bash scripts/run_layer_c_r32.sh
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
R32_DATA_DIR="${R32_DATA_DIR:-/Users/ericmao/Downloads/r3.2}"
OUT_EPISODES="${OUT_EPISODES:-outputs/episodes/cert_r32}"
LIMIT="${LIMIT:-5}"
RUN_ID="${RUN_ID:-cert-r32-1}"

if [[ ! -d "$R32_DATA_DIR" ]]; then
  echo "CERT r3.2 directory not found: $R32_DATA_DIR"
  echo "Set R32_DATA_DIR to your r3.2 path, e.g. export R32_DATA_DIR=/path/to/r3.2"
  exit 1
fi

echo "[1/2] cert2episodes from r3.2 -> $OUT_EPISODES (run_id=$RUN_ID)"
python3 -m src.cli cert2episodes --data_dir "$R32_DATA_DIR" --out_dir "$OUT_EPISODES" --window_days 7 --run_id "$RUN_ID"

echo "[2/2] eval (Layer C: retrieve -> analyze -> writeback) on first $LIMIT episodes"
python3 -m src.cli eval --episodes_dir "$OUT_EPISODES" --limit "$LIMIT"

echo "Done. Metrics: outputs/eval/metrics.csv"
