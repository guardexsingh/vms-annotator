#!/usr/bin/env bash
set -euo pipefail
readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly VENV_PYTHON="$ROOT/.venv/bin/python"
readonly DURATION="${1:-30}"
readonly CONFIG="${2:-config/cameras.relay.yaml}"
readonly OUTPUT="$ROOT/logs/ai_capture_$(date +%Y%m%d_%H%M%S).ndjson"
cd "$ROOT"

cleanup() {
  ./scripts/stop.sh >/dev/null || true
}
trap cleanup EXIT INT TERM
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
export PYTHONNOUSERSITE=1
./scripts/start.sh "$CONFIG"
for strategy in continuous-scaled sampled-scaled continuous-full sampled-full hardware-sampled; do
  "$VENV_PYTHON" -m app.capture_benchmark \
    --config "$CONFIG" --strategy "$strategy" --duration "$DURATION" | tee -a "$OUTPUT"
done
echo "AI capture benchmark: $OUTPUT"
