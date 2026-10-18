#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${PORT:-8000}"

listener_pids() {
  ss -ltnp "sport = :$PORT" | grep -oP 'pid=\K[0-9]+' | sort -u
}

PIDS="$(listener_pids)"

if [ -z "$PIDS" ]; then
  echo "[stop] no listener on port $PORT, nothing to do"
  exit 0
fi

echo "[stop] stopping pids on port $PORT: $(echo "$PIDS" | tr '\n' ' ')"
for pid in $PIDS; do
  kill "$pid"
done

for _ in $(seq 1 15); do
  if [ -z "$(listener_pids)" ]; then
    echo "[stop] stopped"
    exit 0
  fi
  sleep 1
done

echo "[stop] graceful stop timed out, sending SIGKILL"
for pid in $(listener_pids); do
  kill -9 "$pid"
done

if [ -z "$(listener_pids)" ]; then
  echo "[stop] stopped (forced)"
  exit 0
fi

echo "[stop] failed to release port $PORT"
exit 1
