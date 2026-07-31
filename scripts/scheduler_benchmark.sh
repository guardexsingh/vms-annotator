#!/usr/bin/env bash
set -euo pipefail
readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly VENV_PYTHON="$ROOT/.venv/bin/python"
readonly MODE="${1:?scheduler mode required: serial, batch, or opportunistic}"
readonly DURATION="${2:-60}"
readonly SOURCE_CONFIG="${3:-config/cameras.yaml}"
readonly WORKERS="${4:-1}"
readonly THREADS="${5:-4}"
readonly CONFIG="$ROOT/run/scheduler_${MODE}.yaml"
cd "$ROOT"
case "$MODE" in
  serial|batch|opportunistic) ;;
  *) echo "Scheduler mode must be serial, batch, or opportunistic." >&2; exit 2 ;;
esac

cleanup() {
  ./scripts/stop.sh >/dev/null || true
  rm -f "$CONFIG"
}
trap cleanup EXIT INT TERM
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
export PYTHONNOUSERSITE=1
"$VENV_PYTHON" -m app.scheduler_benchmark_config \
  --input "$SOURCE_CONFIG" --output "$CONFIG" --batch-mode "$MODE" \
  --threads "$THREADS" --inference-workers "$WORKERS" --capture-fps 5
./scripts/start.sh "$CONFIG"
./scripts/benchmark.sh "$DURATION"
