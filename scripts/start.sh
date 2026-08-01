#!/usr/bin/env bash
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly _START_SAFE_PATH="$PATH"
# An explicit command-line environment assignment must win over a value in
# .env, while .env remains a supported way to set the optional safe override.
if [[ -v BYTETRACK_PREDICTION_FPS ]]; then
  readonly _START_PARENT_BYTETRACK_PREDICTION_FPS="$BYTETRACK_PREDICTION_FPS"
  readonly _START_PARENT_BYTETRACK_PREDICTION_FPS_SET=1
else
  readonly _START_PARENT_BYTETRACK_PREDICTION_FPS_SET=0
fi
for _START_CONTROL_NAME in DETECTION_BACKEND DETECTION_PRECISION YOLO_INFERENCE_FPS AI_CAPTURE_FPS TRT_ENGINE_MODEL; do
  if [[ -v "$_START_CONTROL_NAME" ]]; then
    printf -v "_START_PARENT_${_START_CONTROL_NAME}" '%s' "${!_START_CONTROL_NAME}"
    printf -v "_START_PARENT_${_START_CONTROL_NAME}_SET" '%s' 1
  else
    printf -v "_START_PARENT_${_START_CONTROL_NAME}_SET" '%s' 0
  fi
done
cd "$SCRIPT_ROOT"

pid_is_alive() {
  [[ "$1" =~ ^[0-9]+$ ]] && kill -0 "$1" 2>/dev/null
}

remove_stale_pid_file() {
  local file="$1" name="$2" pid
  [[ -f "$file" ]] || return 0
  pid="$(<"$file")"
  if ! pid_is_alive "$pid"; then
    rm -f "$file"
    echo "Removed stale $name PID file." >&2
  fi
}

load_dotenv() {
  if [[ ! -f .env ]]; then
    echo "Missing .env" >&2
    return 1
  fi

  local mode group_world_bits
  mode="$(stat -c '%a' .env)"
  group_world_bits=$(( 8#$mode & 8#044 ))
  if (( group_world_bits != 0 )); then
    echo ".env is group-readable or world-readable; run: chmod 600 .env" >&2
    return 1
  fi

  set -a
  # shellcheck disable=SC1091
  if ! source .env; then
    set +a
    echo "Unable to load .env" >&2
    return 1
  fi
  set +a

  if (( _START_PARENT_BYTETRACK_PREDICTION_FPS_SET )); then
    export BYTETRACK_PREDICTION_FPS="$_START_PARENT_BYTETRACK_PREDICTION_FPS"
  fi
  for _START_CONTROL_NAME in DETECTION_BACKEND DETECTION_PRECISION YOLO_INFERENCE_FPS AI_CAPTURE_FPS TRT_ENGINE_MODEL; do
    _START_CONTROL_SET="_START_PARENT_${_START_CONTROL_NAME}_SET"
    _START_CONTROL_VALUE="_START_PARENT_${_START_CONTROL_NAME}"
    if (( ${!_START_CONTROL_SET} )); then
      export "$_START_CONTROL_NAME=${!_START_CONTROL_VALUE}"
    fi
  done

}

validate_camera_environment() {
  local variable_names variable missing=0
  if ! variable_names="$("$VENV_PYTHON" -c '
from app.config import required_camera_environment_variables
import sys
print("\n".join(required_camera_environment_variables(sys.argv[1])))
' "$CAMERA_CONFIG")"; then
    echo "Unable to read camera configuration." >&2
    return 1
  fi
  while IFS= read -r variable; do
    [[ -n "$variable" ]] || continue
    if [[ -z "${!variable:-}" ]]; then
      echo "$variable" >&2
      missing=1
    fi
  done <<< "$variable_names"
  (( missing == 0 ))
}

started_mediamtx_pid=""
started_app_pid=""
cleanup_failed_start() {
  local pid
  for pid in "$started_app_pid" "$started_mediamtx_pid"; do
    [[ -n "$pid" ]] || continue
    if pid_is_alive "$pid"; then
      kill "$pid" 2>/dev/null || true
      for _ in {1..20}; do
        pid_is_alive "$pid" || break
        sleep 0.1
      done
      pid_is_alive "$pid" && kill -KILL "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  [[ -n "$started_app_pid" ]] && rm -f run/app.pid
  [[ -n "$started_mediamtx_pid" ]] && rm -f run/mediamtx.pid
  rm -f "$ROOT/run/mediamtx_direct.yml" "$ROOT/run/video-validation.json"
}

fail_start() {
  echo "Startup failed. See logs/app.log and logs/mediamtx.log." >&2
  cleanup_failed_start
  exit 1
}

if ! load_dotenv; then
  exit 1
fi
PATH="$_START_SAFE_PATH"
export PATH
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
export PYTHONNOUSERSITE=1
readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly VENV_PYTHON="$ROOT/.venv/bin/python"
readonly CAMERA_CONFIG="${1:-config/cameras.yaml}"
readonly MEDIAMTX_CONFIG="${2:-config/mediamtx.yml}"
readonly DIRECT_MEDIAMTX_CONFIG="$ROOT/run/mediamtx_direct.yml"
readonly VIDEO_VALIDATION="$ROOT/run/video-validation.json"
readonly APP_HEALTH_URL="http://127.0.0.1:18080/healthz"
readonly MEDIAMTX_HEALTH_URL="http://127.0.0.1:19997/v3/config/global/get"
readonly HEALTH_TIMEOUT_SECONDS=30
readonly HEALTH_POLL_INTERVAL=0.2
readonly MPLCONFIGDIR="$ROOT/run/matplotlib"
readonly YOLO_CONFIG_DIR="$ROOT/run/ultralytics"
readonly XDG_CONFIG_HOME="$ROOT/run/xdg-config"
export MPLCONFIGDIR YOLO_CONFIG_DIR XDG_CONFIG_HOME
cd "$ROOT"

if [[ ! -f "$CAMERA_CONFIG" ]]; then
  echo "Camera configuration not found: $CAMERA_CONFIG" >&2
  exit 1
fi
if [[ ! -f "$MEDIAMTX_CONFIG" ]]; then
  echo "MediaMTX configuration not found: $MEDIAMTX_CONFIG" >&2
  exit 1
fi
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Isolated Python not found at $ROOT/.venv/bin/python; install requirements-dev.txt first." >&2
  exit 1
fi
if ! "$VENV_PYTHON" -c 'import psutil; import yaml; import cv2; import lap; from ultralytics.trackers.byte_tracker import BYTETracker'; then
  echo "Isolated Python dependency preflight failed." >&2
  exit 1
fi
if ! validate_camera_environment; then
  exit 1
fi
"$VENV_PYTHON" -c 'import sys; print(f"Python executable: {sys.executable}"); print(f"Python prefix: {sys.prefix}")'
mkdir -p logs run
readonly MEDIAMTX_EXECUTABLE="$(command -v mediamtx || true)"
if [[ -z "$MEDIAMTX_EXECUTABLE" ]]; then echo "MediaMTX not found in the startup PATH." >&2; exit 1; fi
remove_stale_pid_file run/mediamtx.pid MediaMTX
remove_stale_pid_file run/app.pid application
if [[ -f run/mediamtx.pid ]] && pid_is_alive "$(<run/mediamtx.pid)"; then echo "MediaMTX already running" >&2; exit 1; fi
if [[ -f run/app.pid ]] && pid_is_alive "$(<run/app.pid)"; then echo "Application already running" >&2; exit 1; fi
readonly VIDEO_MODE="$("$VENV_PYTHON" -m app.direct_video mode --config "$CAMERA_CONFIG")"
runtime_mediamtx_config="$MEDIAMTX_CONFIG"
if [[ "$VIDEO_MODE" == "direct_hevc" ]]; then
  "$VENV_PYTHON" -m app.direct_video generate \
    --config "$CAMERA_CONFIG" --base "$MEDIAMTX_CONFIG" --output "$DIRECT_MEDIAMTX_CONFIG"
  runtime_mediamtx_config="$DIRECT_MEDIAMTX_CONFIG"
fi
readonly RUNTIME_MEDIAMTX_CONFIG="$runtime_mediamtx_config"
"$MEDIAMTX_EXECUTABLE" "$RUNTIME_MEDIAMTX_CONFIG" >logs/mediamtx.log 2>&1 &
started_mediamtx_pid=$!
echo "$started_mediamtx_pid" >run/mediamtx.pid
mkdir -p "$MPLCONFIGDIR" "$YOLO_CONFIG_DIR" "$XDG_CONFIG_HOME"

media_deadline=$((SECONDS + 15))
while (( SECONDS < media_deadline )); do
  pid_is_alive "$started_mediamtx_pid" || fail_start
  if curl --fail --silent --show-error --max-time 1 "$MEDIAMTX_HEALTH_URL" >/dev/null 2>&1; then
    break
  fi
  sleep "$HEALTH_POLL_INTERVAL"
done
if ! curl --fail --silent --show-error --max-time 1 "$MEDIAMTX_HEALTH_URL" >/dev/null 2>&1; then
  fail_start
fi

if [[ "$VIDEO_MODE" == "direct_hevc" ]]; then
  if ! "$VENV_PYTHON" -m app.direct_video verify \
    --config "$CAMERA_CONFIG" --output "$VIDEO_VALIDATION"; then
    fail_start
  fi
fi

nice -n 5 "$VENV_PYTHON" -m app.main --config "$CAMERA_CONFIG" \
  --video-validation "$VIDEO_VALIDATION" >logs/app.log 2>&1 &
started_app_pid=$!
echo "$started_app_pid" >run/app.pid

if ! [[ "$HEALTH_TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] || (( HEALTH_TIMEOUT_SECONDS < 1 )); then
  echo "HEALTH_TIMEOUT_SECONDS must be a positive integer." >&2
  fail_start
fi
deadline=$((SECONDS + HEALTH_TIMEOUT_SECONDS))
while (( SECONDS < deadline )); do
  if ! pid_is_alive "$started_mediamtx_pid" || ! pid_is_alive "$started_app_pid"; then
    fail_start
  fi
  if curl --fail --silent --show-error --max-time 1 "$APP_HEALTH_URL" >/dev/null 2>&1 \
    && pid_is_alive "$started_mediamtx_pid" && pid_is_alive "$started_app_pid"; then
    echo "Started experiment: http://${PUBLIC_HOST:-127.0.0.1}:18080"
    exit 0
  fi
  sleep "$HEALTH_POLL_INTERVAL"
done
fail_start
