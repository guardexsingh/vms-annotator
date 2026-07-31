#!/usr/bin/env bash
set -euo pipefail
readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly VENV_PYTHON="$ROOT/.venv/bin/python"
readonly MPLCONFIGDIR="$ROOT/run/matplotlib"
readonly YOLO_CONFIG_DIR="$ROOT/run/ultralytics"
readonly XDG_CONFIG_HOME="$ROOT/run/xdg-config"
cd "$ROOT"
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
export PYTHONNOUSERSITE=1 MPLCONFIGDIR YOLO_CONFIG_DIR XDG_CONFIG_HOME
mkdir -p "$MPLCONFIGDIR" "$YOLO_CONFIG_DIR" "$XDG_CONFIG_HOME"
exec "$VENV_PYTHON" -m app.main --config config/cameras.yaml --detector-only
