#!/usr/bin/env bash
# Start the MetaDrive demo and expose it with a Cloudflare Quick Tunnel so
# GitHub Pages can reach it over wss://.
#
# Usage:
#   bash tools/run_live_demo_public.sh
#   bash tools/run_live_demo_public.sh --port 8765
#
# Then open the printed GitHub Pages link (includes ?ws=…).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PORT=8765
PAGES_URL="${TRAFFIC_BENCH_PAGES_URL:-https://emb-ai.github.io/traffic-sign-bench/}"
CLOUDFLARED_BIN="${CLOUDFLARED_BIN:-}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="${2:?}"
      shift 2
      ;;
    --pages-url)
      PAGES_URL="${2:?}"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

mkdir -p "$REPO_ROOT/.tools"
if [[ -z "$CLOUDFLARED_BIN" ]]; then
  if command -v cloudflared >/dev/null 2>&1; then
    CLOUDFLARED_BIN="$(command -v cloudflared)"
  else
    CLOUDFLARED_BIN="$REPO_ROOT/.tools/cloudflared"
  fi
fi

if [[ ! -x "$CLOUDFLARED_BIN" ]]; then
  echo "[public-demo] downloading cloudflared → $CLOUDFLARED_BIN"
  curl -fsSL \
    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64" \
    -o "$CLOUDFLARED_BIN"
  chmod +x "$CLOUDFLARED_BIN"
fi

export PYTHONPATH="$REPO_ROOT/third_party/metadrive:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
export SDL_AUDIODRIVER="${SDL_AUDIODRIVER:-dummy}"
export PULSE_SERVER="${PULSE_SERVER:-none}"

LOG_DIR="${TMPDIR:-/tmp}/traffic-bench-demo"
mkdir -p "$LOG_DIR"
DEMO_LOG="$LOG_DIR/demo.log"
TUNNEL_LOG="$LOG_DIR/tunnel.log"
URL_FILE="$LOG_DIR/public_url.txt"
rm -f "$URL_FILE" "$TUNNEL_LOG"

cleanup() {
  [[ -n "${TUNNEL_PID:-}" ]] && kill "$TUNNEL_PID" 2>/dev/null || true
  [[ -n "${DEMO_PID:-}" ]] && kill "$DEMO_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[public-demo] starting MetaDrive backend on 0.0.0.0:${PORT}"
python -m traffic_bench.demo.server \
  --host 0.0.0.0 \
  --port "$PORT" \
  --allowed-origin https://emb-ai.github.io \
  --allow-any-origin \
  "${EXTRA_ARGS[@]}" \
  >"$DEMO_LOG" 2>&1 &
DEMO_PID=$!

for _ in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:${PORT}/api/health" >/dev/null; then
    break
  fi
  if ! kill -0 "$DEMO_PID" 2>/dev/null; then
    echo "[public-demo] demo server exited early; see $DEMO_LOG" >&2
    exit 1
  fi
  sleep 1
done
curl -sf "http://127.0.0.1:${PORT}/api/health" >/dev/null \
  || { echo "[public-demo] demo did not become healthy; see $DEMO_LOG" >&2; exit 1; }

echo "[public-demo] opening Cloudflare Quick Tunnel → http://127.0.0.1:${PORT}"
"$CLOUDFLARED_BIN" tunnel --url "http://127.0.0.1:${PORT}" --no-autoupdate \
  >"$TUNNEL_LOG" 2>&1 &
TUNNEL_PID=$!

PUBLIC_HTTPS=""
for _ in $(seq 1 60); do
  PUBLIC_HTTPS="$(
    grep -Eo 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' "$TUNNEL_LOG" | tail -n 1 || true
  )"
  if [[ -n "$PUBLIC_HTTPS" ]]; then
    break
  fi
  if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
    echo "[public-demo] cloudflared exited early; see $TUNNEL_LOG" >&2
    exit 1
  fi
  sleep 1
done

if [[ -z "$PUBLIC_HTTPS" ]]; then
  echo "[public-demo] could not parse trycloudflare URL; see $TUNNEL_LOG" >&2
  exit 1
fi

WS_BASE="${PUBLIC_HTTPS}/api/drive"
WS_ENC="$(python -c 'import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1], safe=""))' "$WS_BASE")"
PAGES_LINK="${PAGES_URL}?ws=${WS_ENC}"
printf '%s\n' "$PUBLIC_HTTPS" >"$URL_FILE"

cat <<EOF

============================================================
Public demo is up.

  Local:        http://127.0.0.1:${PORT}
  Tunnel:       ${PUBLIC_HTTPS}
  WebSocket:    ${WS_BASE}

Open GitHub Pages with the backend wired in:

  ${PAGES_LINK}

Keep this process running (Ctrl-C stops demo + tunnel).
Logs: $DEMO_LOG
      $TUNNEL_LOG
============================================================

EOF

wait
