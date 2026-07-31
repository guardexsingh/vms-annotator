#!/usr/bin/env bash
set -euo pipefail
readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly VENV_PYTHON="$ROOT/.venv/bin/python"
readonly SCOPE="${1:?scope required: all or an enabled camera ID}"
readonly DURATION="${2:-1800}"
readonly SOURCE_CONFIG="${3:-config/cameras.yaml}"
readonly CONFIG="$ROOT/run/stability_${SCOPE}.yaml"
cd "$ROOT"
cleanup() {
  ./scripts/stop.sh >/dev/null || true
  rm -f "$CONFIG"
}
trap cleanup EXIT INT TERM
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
export PYTHONNOUSERSITE=1
"$VENV_PYTHON" -m app.stability_config \
  --input "$SOURCE_CONFIG" --output "$CONFIG" --scope "$SCOPE"
./scripts/start.sh "$CONFIG"
./scripts/benchmark.sh "$DURATION"
