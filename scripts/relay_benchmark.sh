#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DURATION="${1:-120}"
CONFIG="config/cameras.relay.yaml"

./scripts/start.sh "$CONFIG"
trap './scripts/stop.sh' EXIT INT TERM
./scripts/benchmark.sh "$DURATION"
