#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CAMERA_ID="${1:?camera ID required}"
MODE="${2:-copy}"
DURATION="${3:-300}"
CONFIG="${4:-config/cameras.relay.yaml}"
./scripts/probe_pipeline.sh "$CONFIG"
exec ./scripts/camera_single_test.sh "$CAMERA_ID" "$DURATION" "$CONFIG" "$MODE"
