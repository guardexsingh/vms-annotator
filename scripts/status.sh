#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
pid_is_alive() {
  [[ "$1" =~ ^[0-9]+$ ]] && kill -0 "$1" 2>/dev/null
}

for name in app mediamtx; do
  file="run/$name.pid"
  if [[ ! -f "$file" ]]; then
    echo "$name: process stopped"
    continue
  fi
  pid="$(<"$file")"
  if ! pid_is_alive "$pid"; then
    echo "$name: PID file is stale (pid $pid); process stopped"
    continue
  fi
  if [[ "$name" == app ]]; then
    url="${APP_HEALTH_URL:-http://127.0.0.1:18080/healthz}"
  else
    url="${MEDIAMTX_API_URL:-http://127.0.0.1:19997/v3/config/global/get}"
  fi
  if curl --fail --silent --show-error --max-time 2 "$url" >/dev/null 2>&1; then
    echo "$name: process running (pid $pid); HTTP service ready"
  else
    echo "$name: process running (pid $pid); HTTP service not ready"
  fi
done
