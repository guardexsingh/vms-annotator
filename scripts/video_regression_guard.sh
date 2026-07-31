#!/usr/bin/env bash
set -euo pipefail
readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly VENV_PYTHON="$ROOT/.venv/bin/python"
readonly STAMP="$(date +%Y%m%d_%H%M%S)"
readonly OUTPUT="${1:-$ROOT/logs/video_guard_${STAMP}.json}"
cd "$ROOT"
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
export PYTHONNOUSERSITE=1
"$VENV_PYTHON" -m app.video_regression_guard --output "$OUTPUT"
