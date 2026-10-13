#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/third_party/metadrive:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
export SDL_AUDIODRIVER="${SDL_AUDIODRIVER:-dummy}"
export PULSE_SERVER="${PULSE_SERVER:-none}"

exec python -m traffic_bench.demo.server \
  --host 0.0.0.0 \
  --max-sessions "${TRAFFIC_BENCH_DEMO_MAX_SESSIONS:-8}" \
  --idle-timeout "${TRAFFIC_BENCH_DEMO_IDLE_TIMEOUT:-120}" \
  "$@"
