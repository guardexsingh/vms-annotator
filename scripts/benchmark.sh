#!/usr/bin/env bash
set -euo pipefail
readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly DURATION="${1:-1800}"
readonly STAMP="$(date +%Y%m%d_%H%M%S)"
readonly OUTPUT="$ROOT/logs/benchmark_${STAMP}.ndjson"
readonly PROCESS_OUTPUT="$ROOT/logs/processes_${STAMP}.txt"
readonly VIDEO_BEFORE="$ROOT/logs/video_guard_${STAMP}_before.json"
readonly VIDEO_AFTER="$ROOT/logs/video_guard_${STAMP}_after.json"
cd "$ROOT"
END=$((SECONDS + DURATION))
mkdir -p logs
snapshot_processes() {
  local pid pid_file joined
  local process_ids=()
  for pid_file in run/app.pid run/mediamtx.pid; do
    if [[ -f "$pid_file" ]]; then
      pid="$(<"$pid_file")"
      [[ "$pid" =~ ^[0-9]+$ ]] && process_ids+=("$pid")
    fi
  done
  if (( ${#process_ids[@]} > 0 )); then
    joined="$(IFS=,; echo "${process_ids[*]}")"
    ps -o pid,ppid,comm,args -p "$joined" --ppid "$joined" >"$PROCESS_OUTPUT" 2>/dev/null || true
  fi
}
snapshot_processes
./scripts/video_regression_guard.sh "$VIDEO_BEFORE"
while (( SECONDS < END )); do
  payload="$(curl -fsS http://127.0.0.1:18080/metrics || true)"
  [[ -n "$payload" ]] && printf '%s\n' "$payload"
  sleep 5
done >"$OUTPUT"
snapshot_processes
./scripts/video_regression_guard.sh "$VIDEO_AFTER"
