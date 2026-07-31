#!/usr/bin/env bash
set -euo pipefail
readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly VENV_PYTHON="$ROOT/.venv/bin/python"
readonly CAMERA_ID="${1:?camera ID required}"
readonly DURATION="${2:-300}"
readonly SOURCE_CONFIG="${3:-config/cameras.yaml}"
readonly TEMP_CONFIG="$ROOT/run/metadata_${CAMERA_ID}.yaml"
cd "$ROOT"
ACTIVE_CONFIG="$TEMP_CONFIG"

cleanup() {
  ./scripts/stop.sh
  rm -f "$TEMP_CONFIG"
}
trap cleanup EXIT INT TERM
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
export PYTHONNOUSERSITE=1
if [[ "$CAMERA_ID" == "all" ]]; then
  ACTIVE_CONFIG="$SOURCE_CONFIG"
else
  "$VENV_PYTHON" -m app.detection_validation_config --input "$SOURCE_CONFIG" --camera "$CAMERA_ID" --output "$TEMP_CONFIG"
fi
readonly ACTIVE_CONFIG
./scripts/start.sh "$ACTIVE_CONFIG"
./scripts/benchmark.sh "$DURATION"
