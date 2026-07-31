#!/usr/bin/env bash
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly _PROBE_SAFE_PATH="$PATH"
cd "$SCRIPT_ROOT"
[[ -f .env ]] || { echo "Missing .env" >&2; exit 1; }
set -a
# shellcheck disable=SC1091
source .env
set +a
PATH="$_PROBE_SAFE_PATH"
export PATH
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
export PYTHONNOUSERSITE=1
readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly VENV_PYTHON="$ROOT/.venv/bin/python"
readonly CONFIG="${1:-config/cameras.relay.yaml}"
exec "$VENV_PYTHON" -m app.probe_pipeline --config "$CONFIG"
