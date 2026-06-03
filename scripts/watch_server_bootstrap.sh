#!/usr/bin/env bash
# Watch a remote serve_cb.sh uvicorn process until it reports ready or dies.
# Emits one line per state change so Monitor (or `tail -f` from a human)
# only sees meaningful progress, not constant polling.
#
# Usage:
#   scripts/watch_server_bootstrap.sh <PID> [HOST] [POLL_SECONDS]
#
# Defaults: HOST=qb1, POLL_SECONDS=10
#
# Emits one of:
#   t+<N>0s alive=<etime>|stage=<bootstrap-log-last-line>|ready=<no|"ready":true>
#   READY
#   PROCESS_GONE  (followed by last 30 server log lines)
#
# This replaces the inline shell-loop monitor we kept re-writing inline.

set -u

PID="${1:-}"
HOST="${2:-qb1}"
POLL="${3:-10}"

if [[ -z "$PID" ]]; then
  echo "usage: $0 <PID> [HOST=qb1] [POLL_SECONDS=10]" >&2
  exit 64
fi

prev_state=""
for i in $(seq 1 200); do
  # Single round-trip to fetch all 3 pieces of state. Newline-separated.
  out=$(ssh "$HOST" "
    stage=\$(tail -1 ~/tt-xla/.cache/server_cb.bootstrap.log 2>/dev/null);
    alive=\$(ps -p $PID -o etime= 2>/dev/null | tr -d ' ');
    health=\$(curl -s --max-time 2 http://localhost:8000/health 2>/dev/null);
    printf '%s\n%s\n%s\n' \"\$stage\" \"\$alive\" \"\$health\"
  " 2>/dev/null)

  stage=$(printf '%s\n' "$out" | sed -n '1p')
  alive=$(printf '%s\n' "$out" | sed -n '2p')
  health=$(printf '%s\n' "$out" | sed -n '3p')
  ready=$(printf '%s\n' "$health" | grep -o '"ready":true' | head -1)

  state="alive=${alive:-GONE}|stage=${stage:-<empty>}|ready=${ready:-no}"
  if [[ "$state" != "$prev_state" ]]; then
    printf 't+%d0s %s\n' "$i" "$state"
    prev_state="$state"
  fi

  if [[ -n "$ready" ]]; then
    echo "READY"
    exit 0
  fi
  if [[ -z "$alive" ]]; then
    echo "PROCESS_GONE"
    ssh "$HOST" 'tail -30 ~/tt-xla/.cache/server_cb.log' | head -60
    exit 1
  fi

  sleep "$POLL"
done

echo "TIMEOUT after ${i}0s; process still alive but never reported ready"
exit 2
