#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CAMERA_ID="${1:?camera ID required}"
MODE="${2:?mode required: copy, transcode, or direct}"
DURATION="${3:-300}"
CONFIG="${4:-config/cameras.relay.yaml}"
case "$MODE" in copy|transcode|direct) ;; *) echo "H.264 mode must be copy, transcode, or direct." >&2; exit 2 ;; esac
exec ./scripts/latency_test.sh "$CAMERA_ID" "$MODE" "$DURATION" "$CONFIG"
