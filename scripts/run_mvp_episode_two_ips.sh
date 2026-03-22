#!/usr/bin/env bash
# AgenticRAG4TimeSeries — 依序對多個 IP 跑 Wazuh Episode + Layer C 完整分析（mvp_wazuh_episode_pg.py）
#
# 預設 IP：192.168.1.203、192.168.1.126
# 環境變數（可選）：
#   MVP_IPS="192.168.1.203 192.168.1.126"   覆寫 IP 清單
#   HOURS=24 SIZE=500                     對應腳本參數
#   MATCH_ALL=1                           預設加 --match-all（Indexer 內常無 IP 欄位時用）；設 0 則依 IP 篩選
#
# 其餘參數會原樣傳給 mvp_wazuh_episode_pg.py，例如：
#   ./scripts/run_mvp_episode_two_ips.sh --skip-db
#   ./scripts/run_mvp_episode_two_ips.sh --no-writeback --hours 48
#
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH=.

HOURS="${HOURS:-24}"
SIZE="${SIZE:-500}"
MATCH_ALL="${MATCH_ALL:-1}"

IPS_STR="${MVP_IPS:-192.168.1.203 192.168.1.126}"
read -r -a IPS <<< "$IPS_STR"

EXTRA=(--hours "$HOURS" --size "$SIZE")
if [[ "$MATCH_ALL" == "1" ]]; then
  EXTRA+=(--match-all)
fi

failed=0
for ip in "${IPS[@]}"; do
  echo ""
  echo "================================================================================"
  echo " MVP Wazuh → Episode → Layer C  |  target-ip=${ip}"
  echo "================================================================================"
  if ! python3 scripts/mvp_wazuh_episode_pg.py --target-ip "$ip" "${EXTRA[@]}" "$@"; then
    echo "[run_mvp_episode_two_ips] FAILED for ${ip}" >&2
    failed=$((failed + 1))
  else
    echo "[run_mvp_episode_two_ips] OK for ${ip}"
  fi
done

echo ""
if [[ "$failed" -gt 0 ]]; then
  echo "[run_mvp_episode_two_ips] Done with ${failed} failure(s)." >&2
  exit 1
fi
echo "[run_mvp_episode_two_ips] All targets succeeded."
exit 0
