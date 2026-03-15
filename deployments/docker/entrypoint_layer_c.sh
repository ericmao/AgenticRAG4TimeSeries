#!/bin/sh
# Wait for Control Plane API then run Layer C demo (retrieve → analyze → writeback).
set -e

BASE_URL="${CONTROL_PLANE_BASE_URL:-}"
if [ -n "$BASE_URL" ]; then
  HEALTH_URL="${BASE_URL}/health"
  echo "Waiting for Control Plane at $HEALTH_URL ..."
  python3 -c "
import urllib.request, time, sys
url = sys.argv[1]
for i in range(10):
  try:
    urllib.request.urlopen(url, timeout=5)
    print('Control Plane is up.')
    break
  except Exception as e:
    if i == 9:
      print('Warning: Control Plane not ready after 30s:', e)
    time.sleep(3)
" "$HEALTH_URL"
fi

EPISODE="${LAYERC_EPISODE_PATH:-tests/demo/episode_insider_highrisk.json}"
echo "=== Layer C demo: $EPISODE ==="

python3 -m src.cli retrieve --episode "$EPISODE" && \
python3 -m src.cli analyze --episode "$EPISODE" && \
python3 -m src.cli writeback --episode "$EPISODE"

echo "Layer C demo done. Writeback posted to Control Plane if CONTROL_PLANE_BASE_URL was set."
