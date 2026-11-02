#!/usr/bin/env bash
# Start the Cloudflare named tunnel that exposes the chat server.
#
# Prerequisites:
#   $CHAT_DIR/tunnel-token  contains the token printed by
#                           `cloudflared tunnel token <name>` (or the API).
#
# The cloudflared binary is cached on the NAS so a recycled container does not
# have to download it again. Downloads go through the gh-proxy mirror and are
# size-checked: a truncated binary segfaults on startup.
set -euo pipefail

CHAT_DIR="${CHAT_DIR:-/mnt/workspace/chat}"
BIN="$CHAT_DIR/bin/cloudflared"
TOKEN_FILE="$CHAT_DIR/tunnel-token"
EXPECTED_SIZE="${CLOUDFLARED_SIZE:-40274445}"
MIRROR="${CLOUDFLARED_MIRROR:-https://gh-proxy.com/https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64}"

if ! curl -fsS "http://127.0.0.1:8899/api/status" >/dev/null 2>&1; then
    echo "[WARN] chat server is not answering on 8899; run setup.sh first"
fi

mkdir -p "$CHAT_DIR/bin"

download() {
    for attempt in 1 2 3; do
        echo "[..] downloading cloudflared (attempt $attempt)"
        if curl -fL --max-time 600 -o "$BIN.tmp" "$MIRROR"; then
            if [ "$(stat -c%s "$BIN.tmp" 2>/dev/null || echo 0)" = "$EXPECTED_SIZE" ]; then
                mv "$BIN.tmp" "$BIN"
                chmod +x "$BIN"
                return 0
            fi
            echo "[WARN] size mismatch: $(stat -c%s "$BIN.tmp" 2>/dev/null || echo 0) != $EXPECTED_SIZE"
        fi
        sleep 3
    done
    return 1
}

if [ ! -x "$BIN" ] || [ "$(stat -c%s "$BIN" 2>/dev/null || echo 0)" != "$EXPECTED_SIZE" ]; then
    download || { echo "[FAIL] could not obtain a valid cloudflared binary"; exit 1; }
fi

if [ ! -f "$TOKEN_FILE" ]; then
    echo "[FAIL] missing $TOKEN_FILE"
    echo "       create a named tunnel in Cloudflare and save its token there"
    exit 1
fi

TOKEN="$(cat "$TOKEN_FILE")"
pkill -f "cloudflared tunnel run" 2>/dev/null || true
sleep 1
nohup "$BIN" tunnel run --token "$TOKEN" > /tmp/cloudflared.log 2>&1 &
sleep 5
tail -n 8 /tmp/cloudflared.log || true
echo "[OK] cloudflared started (log: /tmp/cloudflared.log)"
