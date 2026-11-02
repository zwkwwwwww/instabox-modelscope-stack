#!/usr/bin/env bash
# Start (or restart) the chat server on the ModelScope CPU instance.
# All persistent files live under $CHAT_DIR, which should be the NAS mount
# (/mnt/workspace) so they survive container recycling.
set -euo pipefail

CHAT_DIR="${CHAT_DIR:-/mnt/workspace/chat}"
CHAT_PORT="${CHAT_PORT:-8899}"

cd "$CHAT_DIR"

# Stop the previous instance without pkill: the job's own command line often
# contains "chat_server.py", so pkill -f can kill the shell that runs it.
if command -v fuser >/dev/null 2>&1; then
    fuser -k "${CHAT_PORT}/tcp" 2>/dev/null || true
fi
sleep 1

# Prefer the NAS wheel cache: it makes recovery work without internet.
if [ -d "$CHAT_DIR/wheels" ]; then
    pip install --no-index --find-links "$CHAT_DIR/wheels" -r "$CHAT_DIR/requirements.txt" \
        || pip install -r "$CHAT_DIR/requirements.txt"
else
    pip install -r "$CHAT_DIR/requirements.txt"
fi

export CHAT_PORT
nohup python3 "$CHAT_DIR/chat_server.py" > /tmp/chat_server.log 2>&1 &

for i in $(seq 1 30); do
    sleep 1
    if curl -fsS "http://127.0.0.1:${CHAT_PORT}/api/status" >/dev/null 2>&1; then
        echo "[OK] chat server listening on ${CHAT_PORT}"
        exit 0
    fi
done

echo "[FAIL] chat server did not come up; last log lines:"
tail -30 /tmp/chat_server.log
exit 1
