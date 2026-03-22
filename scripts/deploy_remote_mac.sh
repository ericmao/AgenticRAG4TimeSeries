#!/usr/bin/env bash
# 將本機 repo rsync 至遠端，並執行 docker compose（docker-compose.local.yml）。
#
# 密碼登入：安裝 sshpass，並 export SSHPASS（勿將密碼寫入版本庫）
#   export SSHPASS='...'
#   DEPLOY_HOST=192.168.1.203 DEPLOY_USER=avocado.ai ./scripts/deploy_remote_mac.sh
#
# 遠端 macOS：非互動 SSH 的 PATH 常不含 /usr/local/bin，已內建補上，避免 docker-credential-desktop 找不到。
#
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_HOST="${DEPLOY_HOST:-192.168.1.203}"
REMOTE_USER="${DEPLOY_USER:-avocado.ai}"
# 遠端目錄（相對於遠端 $HOME，可改）
REMOTE_SUBDIR="${DEPLOY_REMOTE_SUBDIR:-AgenticRAG4TimeSeries}"
COMPOSE_FILE="${DEPLOY_COMPOSE_FILE:-docker-compose.local.yml}"

RSYNC_EXCLUDES=(
  --exclude 'node_modules'
  --exclude '**/node_modules'
  --exclude '__pycache__'
  --exclude '.git'
  --exclude '.pytest_cache'
  --exclude '*.pyc'
  --exclude '.env'
)
# rsync 不覆寫遠端 .env。Wazuh「手動分析」需遠端 repo 根目錄 .env 含 WAZUH_INDEXER_*（見 deployments/dotenv.wazuh.example）。

if [[ -n "${SSHPASS:-}" ]] && command -v sshpass >/dev/null 2>&1; then
  export SSHPASS
  RSYNC=(sshpass -e rsync -az -e "ssh -o StrictHostKeyChecking=accept-new")
  SSH=(sshpass -e ssh -o StrictHostKeyChecking=accept-new)
else
  RSYNC=(rsync -az -e "ssh -o StrictHostKeyChecking=accept-new")
  SSH=(ssh -o StrictHostKeyChecking=accept-new)
fi

echo "+ rsync -> ${REMOTE_USER}@${REMOTE_HOST}:~/${REMOTE_SUBDIR}/"
"${RSYNC[@]}" "${RSYNC_EXCLUDES[@]}" "$REPO_ROOT/" "${REMOTE_USER}@${REMOTE_HOST}:~/${REMOTE_SUBDIR}/"

echo "+ docker compose on remote"
"${SSH[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
  "export PATH=\"/usr/local/bin:/usr/bin:/bin:\${PATH:-}\"; cd \"\$HOME/${REMOTE_SUBDIR}\" && docker compose -f ${COMPOSE_FILE} up -d --build"

echo "done. MVP UI: http://${REMOTE_HOST}:8765"
