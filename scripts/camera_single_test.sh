#!/usr/bin/env bash
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly _SINGLE_TEST_SAFE_PATH="$PATH"
cd "$SCRIPT_ROOT"

# This helper needs the selected URL only to write a short-lived, private
# MediaMTX direct-pull file.  Re-establish every execution control afterwards.
[[ -f .env ]] || { echo "Missing .env" >&2; exit 1; }
set -a
# shellcheck disable=SC1091
source .env
set +a
PATH="$_SINGLE_TEST_SAFE_PATH"
export PATH
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
export PYTHONNOUSERSITE=1
readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly VENV_PYTHON="$ROOT/.venv/bin/python"
readonly CAMERA_ID="${1:?camera ID required}"
readonly DURATION="${2:-300}"
readonly CONFIG="${3:-config/cameras.relay.yaml}"
readonly H264_MODE="${4:-copy}"
case "$H264_MODE" in copy|transcode|direct) ;; *) echo "H.264 mode must be copy, transcode, or direct." >&2; exit 2 ;; esac
readonly APP_CONFIG="$ROOT/run/single_${CAMERA_ID}_${H264_MODE}.yaml"
readonly MEDIAMTX_CONFIG="$ROOT/run/mediamtx_${CAMERA_ID}_${H264_MODE}.yml"

cleanup() {
  ./scripts/stop.sh
  rm -f "$APP_CONFIG" "$MEDIAMTX_CONFIG"
}
trap cleanup EXIT INT TERM
"$VENV_PYTHON" -m app.test_config --input "$CONFIG" --camera "$CAMERA_ID" --h264-mode "$H264_MODE" \
  --app-output "$APP_CONFIG" --mediamtx-output "$MEDIAMTX_CONFIG"
./scripts/start.sh "$APP_CONFIG" "$MEDIAMTX_CONFIG"
./scripts/benchmark.sh "$DURATION"
