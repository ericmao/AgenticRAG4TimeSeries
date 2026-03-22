#!/usr/bin/env bash
# 將 cert_r32 目錄內前 N 個 Episode 跑 mvp_cert_layer_c 並寫入 PostgreSQL，供 http://localhost:8765/runs 顯示。
# 需 DATABASE_URL（與 MVP UI 相同），例如本機 Docker：
#   export DATABASE_URL=postgresql://agentic:agentic@127.0.0.1:5435/agentic
#
# Usage:
#   DATABASE_URL=... bash scripts/r32_episodes_to_analysis_runs.sh
#   EPISODES_DIR=outputs/episodes/cert_r32 LIMIT=10 DATABASE_URL=... bash scripts/r32_episodes_to_analysis_runs.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${PYTHONPATH:-.}"

EPISODES_DIR="${EPISODES_DIR:-outputs/episodes/cert_r32}"
LIMIT="${LIMIT:-5}"

if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "請設定 DATABASE_URL（與 mvp_ui 連線同一 Postgres），例如：" >&2
  echo "  export DATABASE_URL=postgresql://agentic:agentic@127.0.0.1:5435/agentic" >&2
  exit 1
fi

if [[ ! -d "$EPISODES_DIR" ]]; then
  echo "目錄不存在: $EPISODES_DIR（請先跑 scripts/run_layer_c_r32.sh 產生 episodes）" >&2
  exit 1
fi

FILES=()
while IFS= read -r line; do
  FILES+=("$line")
done < <(find "$EPISODES_DIR" -maxdepth 1 -name '*.json' -type f | sort | head -n "$LIMIT")

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "找不到任何 episode JSON：$EPISODES_DIR" >&2
  exit 1
fi

echo "將寫入 analysis_runs（共 ${#FILES[@]} 筆，LIMIT=$LIMIT）..."
for f in "${FILES[@]}"; do
  echo "--- $f"
  python3 scripts/mvp_cert_layer_c.py --episode "$f" || true
done
echo "完成。請開 http://localhost:8765/runs 查看（source=cert）。"
