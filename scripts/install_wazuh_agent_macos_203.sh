#!/usr/bin/env bash
# 在 macOS（Apple Silicon）上安裝 Wazuh Agent 4.14.4，並指向 Manager 192.168.1.203
# 用法：chmod +x scripts/install_wazuh_agent_macos_203.sh && ./scripts/install_wazuh_agent_macos_203.sh

set -euo pipefail
MANAGER="${WAZUH_MANAGER:-192.168.1.203}"
VER="4.14.4"
PKG_URL="https://packages.wazuh.com/4.x/macos/wazuh-agent-${VER}-1.arm64.pkg"
PKG="/tmp/wazuh-agent-${VER}-1.arm64.pkg"

echo ">>> Manager: $MANAGER"
echo ">>> Download: $PKG_URL"
curl -fsSL -o "$PKG" "$PKG_URL"
echo "WAZUH_MANAGER='$MANAGER'" > /tmp/wazuh_envs
echo ">>> Install (需要輸入本機密碼)..."
sudo installer -pkg "$PKG" -target /
echo ">>> Start agent"
sudo launchctl bootstrap system /Library/LaunchDaemons/com.wazuh.agent.plist 2>/dev/null || sudo launchctl kickstart -k system/com.wazuh.agent 2>/dev/null || true
sleep 3
sudo /Library/Ossec/bin/wazuh-control status
echo ">>> ossec.conf address:"
sudo grep '<address>' /Library/Ossec/etc/ossec.conf
echo ">>> Recent agentd log:"
sudo grep 'wazuh-agentd' /Library/Ossec/logs/ossec.log | tail -12
