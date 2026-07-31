#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

app_url="${APP_HEALTH_URL:-http://127.0.0.1:18080/healthz}"
mediamtx_url="${MEDIAMTX_API_URL:-http://127.0.0.1:19997/v3/config/global/get}"
failed=0
if curl --fail --silent --show-error --max-time 3 "$app_url" >/dev/null; then
  echo "application health: ready"
else
  echo "application health: unavailable" >&2
  failed=1
fi
if curl --fail --silent --show-error --max-time 3 "$mediamtx_url" >/dev/null; then
  echo "MediaMTX API: ready"
else
  echo "MediaMTX API: unavailable" >&2
  failed=1
fi
exit "$failed"
