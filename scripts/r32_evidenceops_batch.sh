#!/usr/bin/env bash
# r3.2 → cert2episodes（可選）→ 對前 N 則跑 EvidenceOps（CaseOrchestrator + agents）
#
# Usage:
#   bash scripts/r32_evidenceops_batch.sh
#   SKIP_CERT2=1 LIMIT=5 bash scripts/r32_evidenceops_batch.sh
#   R32_DATA_DIR=/path/to/r3.2 LIMIT=10 bash scripts/r32_evidenceops_batch.sh
#
# Env:
#   R32_DATA_DIR   — CERT r3.2 目錄（預設 ~/Downloads/r3.2）
#   OUT_EPISODES   — episode 輸出目錄（預設 outputs/episodes/cert_r32）
#   RUN_ID         — cert2episodes 用 run_id
#   LIMIT          — 只對排序後前 N 則跑 orchestrator（預設 5）
#   SKIP_CERT2     — 設為 1 略過 cert2episodes（已有 episode JSON 時）
#   SKIP_WRITEBACK — 1 時傳給 orchestrator --skip-writeback（預設 1 以加快批次）
#   SKIP_DEMO      — 1 時傳 --skip-demo-report（預設 1）
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${PYTHONPATH:-.}"

R32_DATA_DIR="${R32_DATA_DIR:-$HOME/Downloads/r3.2}"
OUT_EPISODES="${OUT_EPISODES:-outputs/episodes/cert_r32}"
RUN_ID="${RUN_ID:-cert-r32-evidenceops-$(date +%Y%m%d%H%M)}"
LIMIT="${LIMIT:-5}"
SKIP_CERT2="${SKIP_CERT2:-0}"
SKIP_WRITEBACK="${SKIP_WRITEBACK:-1}"
SKIP_DEMO="${SKIP_DEMO:-1}"

if [[ "$SKIP_CERT2" != "1" ]]; then
  if [[ ! -d "$R32_DATA_DIR" ]]; then
    echo "CERT r3.2 目錄不存在: $R32_DATA_DIR" >&2
    echo "請設定 R32_DATA_DIR，或 SKIP_CERT2=1 使用既有 $OUT_EPISODES" >&2
    exit 1
  fi
  echo "[1/2] cert2episodes: data_dir=$R32_DATA_DIR -> $OUT_EPISODES run_id=$RUN_ID"
  python3 -m src.cli cert2episodes --data_dir "$R32_DATA_DIR" --out_dir "$OUT_EPISODES" --window_days 7 --run_id "$RUN_ID"
else
  echo "[1/2] skip cert2episodes (SKIP_CERT2=1)，使用既有: $OUT_EPISODES"
fi

if [[ ! -d "$OUT_EPISODES" ]]; then
  echo "目錄不存在: $OUT_EPISODES" >&2
  exit 1
fi

FILES=()
while IFS= read -r line; do
  FILES+=("$line")
done < <(find "$OUT_EPISODES" -maxdepth 1 -name '*.json' -type f | sort | head -n "$LIMIT")

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "找不到 episode JSON：$OUT_EPISODES" >&2
  exit 1
fi

echo "[2/2] EvidenceOps orchestrator：共 ${#FILES[@]} 則（LIMIT=$LIMIT）"
ok=0
fail=0
for f in "${FILES[@]}"; do
  rel="${f#$REPO_ROOT/}"
  echo "=== $rel ==="
  extra=()
  [[ "$SKIP_WRITEBACK" == "1" ]] && extra+=(--skip-writeback)
  [[ "$SKIP_DEMO" == "1" ]] && extra+=(--skip-demo-report)
  if python3 scripts/run_layerc_case_orchestrator.py --episode "$f" "${extra[@]}"; then
    ok=$((ok + 1))
  else
    fail=$((fail + 1))
    echo "FAILED: $rel" >&2
  fi
done

echo "Done. ok=$ok fail=$fail"
echo "  outputs/evidenceops/decision_bundle_*.json"
echo "  outputs/agent_activity.json（最後一則覆寫；看 /agents）"
exit 0
