#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
APP="${APP:-app.main:app}"
LOG_FILE="$SCRIPT_DIR/storage/uvicorn.log"
PID_FILE="$SCRIPT_DIR/storage/uvicorn.pid"

mkdir -p "$SCRIPT_DIR/storage"

listener_pids() {
  ss -ltnp "sport = :$PORT" | grep -oP 'pid=\K[0-9]+' | sort -u
}

if [ -n "$(listener_pids)" ]; then
  echo "[start] already running on port $PORT (pid: $(listener_pids | tr '\n' ' '))"
  exit 0
fi

# Python 服务无编译产物；若日后引入前端构建，请在此清理对应缓存目录
export PYTHONUNBUFFERED=1
export PORT

nohup uvicorn "$APP" --host "$HOST" --port "$PORT" > "$LOG_FILE" 2>&1 &
NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"

for _ in $(seq 1 30); do
  if [ -n "$(listener_pids)" ]; then
    echo "[start] started on port $PORT (pid: $(listener_pids | tr '\n' ' '))"
    exit 0
  fi
  if ! kill -0 "$NEW_PID"; then
    echo "[start] process exited during startup, see $LOG_FILE"
    exit 2
  fi
  sleep 1
done

echo "[start] failed to become ready within 30s, see $LOG_FILE"
exit 2
