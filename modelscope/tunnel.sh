#!/bin/bash
# 一键恢复：聊天服务 + 单个 Cloudflare 隧道
# 用法：bash tunnel.sh          已有连接器则直接退出
#       bash tunnel.sh --restart 杀掉旧连接器并重启
set -u

CHAT_DIR=/mnt/workspace/chat
CF="$CHAT_DIR/bin/cloudflared"
TOKEN_FILE="$CHAT_DIR/tunnel-token"
LOCK="$CHAT_DIR/tunnel.lock"
EXPECT=40274445
PAT="cloudflared tunnel run"
QPAT="cloudflared tunnel --url"

# 1) 单实例锁：防止两次 tunnel.sh 并发启动
if [ -f "$LOCK" ]; then
  OLD=$(cat "$LOCK" 2>/dev/null)
  if [ -n "$OLD" ] && kill -0 "$OLD" 2>/dev/null; then
    echo "[SKIP] tunnel.sh 已在运行 (PID $OLD)"
    exit 0
  fi
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT INT TERM

# 2) 已有 cloudflared 就不重复启动
RUNNING=$(pgrep -f "$PAT" | head -n1)
QRUNNING=$(pgrep -f "$QPAT" | head -n1)
if [ -n "$RUNNING" ] || [ -n "$QRUNNING" ]; then
  if [ "${1:-}" = "--restart" ]; then
    echo "[..] 停止旧 cloudflared"
    pkill -9 -f "$PAT" 2>/dev/null || true
    pkill -9 -f "$QPAT" 2>/dev/null || true
    sleep 2
  else
    echo "[OK] cloudflared 已在运行 (PID ${RUNNING:-$QRUNNING})；需要重启用 --restart"
    exit 0
  fi
fi

# 3) 聊天服务
echo '[1/3] 检查聊天服务...'
curl -s --max-time 3 localhost:8899/api/status >/dev/null 2>&1 || bash "$CHAT_DIR/setup.sh"

# 4) cloudflared 二进制
echo '[2/3] cloudflared...'
if ! [ -x "$CF" ] || [ "$(stat -c%s "$CF" 2>/dev/null || echo 0)" != "$EXPECT" ]; then
  echo 'NAS 上没有完好的 cloudflared，从镜像下载并校验...'
  mkdir -p "$CHAT_DIR/bin"
  ok=0
  for i in 1 2 3; do
    curl -sL --max-time 900 -o "$CF" "https://gh-proxy.com/https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    S=$(stat -c%s "$CF" 2>/dev/null || echo 0)
    echo "第${i}次: $S bytes (需要 $EXPECT)"
    [ "$S" = "$EXPECT" ] && { ok=1; break; }
    rm -f "$CF"; sleep 2
  done
  [ "$ok" = 0 ] && { echo 'cloudflared 下载失败'; exit 1; }
  chmod +x "$CF"
fi
"$CF" --version || { echo 'cloudflared 无法执行'; exit 1; }

# 5) 启动且只保留一个连接器
echo '[3/3] 启动隧道...'
if [ -f "$TOKEN_FILE" ]; then
  TOK=$(cat "$TOKEN_FILE")
  nohup "$CF" tunnel run --token "$TOK" >/tmp/cf.log 2>&1 &
  sleep 6
  N=$(pgrep -f "$PAT" | wc -l)
  echo "[OK] 命名隧道已启动，连接器数量: $N"
  [ "$N" -gt 1 ] && echo '[WARN] 检测到多个连接器，请执行 bash tunnel.sh --restart'
  tail -3 /tmp/cf.log
else
  echo '未找到 token，退回快速隧道...'
  nohup "$CF" tunnel --url http://127.0.0.1:8899 --no-autoupdate >/tmp/cf.log 2>&1 &
  U=''
  for i in $(seq 1 30); do
    U=$(grep -oE 'https://[a-z0-9-]+[.]trycloudflare[.]com' /tmp/cf.log 2>/dev/null | head -1)
    [ -n "$U" ] && break
    sleep 2
  done
  echo "CF_URL=$U"
fi
