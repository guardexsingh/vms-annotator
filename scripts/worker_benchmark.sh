#!/usr/bin/env bash
set -euo pipefail
readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly VENV_PYTHON="$ROOT/.venv/bin/python"
readonly DURATION="${1:-20}"
readonly CONFIG="${2:-config/cameras.yaml}"
readonly OUTPUT="$ROOT/logs/workers_$(date +%Y%m%d_%H%M%S).jsonl"
cd "$ROOT"
cleanup() {
  ./scripts/stop.sh >/dev/null || true
}
trap cleanup EXIT INT TERM
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
export PYTHONNOUSERSITE=1
./scripts/start.sh config/cameras.relay.yaml
./scripts/video_regression_guard.sh
"$VENV_PYTHON" -m app.worker_benchmark \
  --config "$CONFIG" --duration "$DURATION" --layout all | tee "$OUTPUT"
./scripts/video_regression_guard.sh
echo "Worker benchmark: $OUTPUT"
