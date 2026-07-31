#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
is_experiment_process() {
  local name="$1" pid="$2" command
  command="$(ps -p "$pid" -o args= 2>/dev/null || true)"
  case "$name" in
    app) [[ "$command" == *"$ROOT/.venv/bin/python"* && "$command" == *"-m app.main"* ]] ;;
    mediamtx) [[ "$command" == *"mediamtx"* && ( "$command" == *"config/mediamtx"* || "$command" == *"run/mediamtx_"* ) ]] ;;
  esac
}

for file in run/app.pid run/mediamtx.pid; do
  [[ -f "$file" ]] || continue
  pid="$(<"$file")"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    name="${file#run/}"; name="${name%.pid}"
    if is_experiment_process "$name" "$pid"; then
      kill "$pid" 2>/dev/null || true
    else
      echo "Refusing to stop unrelated process recorded in $file." >&2
    fi
  fi
  rm -f "$file"
done
rm -f "$ROOT/run/mediamtx_direct.yml" "$ROOT/run/video-validation.json"
echo "Experiment stopped."
