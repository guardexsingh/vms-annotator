#!/usr/bin/env bash
set -euo pipefail
readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly VENV_PYTHON="$ROOT/.venv/bin/python"
readonly ITERATIONS="${1:-15}"
readonly CONFIG="${2:-config/cameras.yaml}"
readonly OUTPUT="$ROOT/logs/inference_$(date +%Y%m%d_%H%M%S).json"
cd "$ROOT"

cleanup() {
  ./scripts/stop.sh >/dev/null || true
}
trap cleanup EXIT INT TERM
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
export PYTHONNOUSERSITE=1
./scripts/start.sh config/cameras.relay.yaml
for threads in 1 2 4 6; do
  OMP_NUM_THREADS="$threads" \
  OPENBLAS_NUM_THREADS="$threads" \
  MKL_NUM_THREADS="$threads" \
  NUMEXPR_NUM_THREADS="$threads" \
  "$VENV_PYTHON" -m app.inference_benchmark \
    --config "$CONFIG" --iterations "$ITERATIONS" --threads "$threads" >>"$OUTPUT"
done
echo "Inference benchmark: $OUTPUT"
