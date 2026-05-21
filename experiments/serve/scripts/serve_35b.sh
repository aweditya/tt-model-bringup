#!/usr/bin/env bash
# Start/stop/restart wrapper for server_35b.py on qb1.
# Mirrors serve.sh / serve_tp.sh patterns from the 27B server.

set -eu
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/tt-xla}"
CACHE_DIR="$PROJECT_ROOT/.cache"
VENV_PY="${VENV_PY:-$PROJECT_ROOT/.venv/bin/python}"
SOCK_FILE="$CACHE_DIR/server_35b.sock"
PID_FILE="$CACHE_DIR/server_35b.pid"
LOG_FILE="$CACHE_DIR/server_35b.log"

cmd_start() {
    cd "$PROJECT_ROOT" || { echo "cannot cd $PROJECT_ROOT"; exit 1; }
    if [ ! -x "$VENV_PY" ]; then
        echo "venv python not found at $VENV_PY"; exit 1
    fi
    mkdir -p "$CACHE_DIR"

    if [ -S "$SOCK_FILE" ] && [ -f "$PID_FILE" ]; then
        if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "already running (pid $(cat "$PID_FILE"))"
            return 0
        fi
        rm -f "$PID_FILE" "$SOCK_FILE"
    fi

    echo "starting server_35b (logs: $LOG_FILE)…"

    TT_METAL_HOME="${TT_METAL_HOME:-$HOME/tenstorrent/tt-metal}"
    TT_BUILD_DIR="${TT_BUILD_DIR:-$TT_METAL_HOME/build_Release}"

    HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" \
    TT_METAL_HOME="$TT_METAL_HOME" \
    ARCH_NAME="${ARCH_NAME:-blackhole}" \
    PYTHONPATH="$TT_METAL_HOME/ttnn:${PYTHONPATH:-}" \
    LD_LIBRARY_PATH="$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib:${LD_LIBRARY_PATH:-}" \
    nohup setsid "$VENV_PY" -m experiments.serve.server_35b \
        >> "$LOG_FILE" 2>&1 &
    pid="$!"
    echo "$pid" > "$PID_FILE"
    echo "launched (pid $pid) — bootstrap takes ~15 s for weight load"
    echo "  tail -f $LOG_FILE"
}

cmd_stop() {
    if [ ! -f "$PID_FILE" ]; then
        if [ -S "$SOCK_FILE" ]; then
            echo "no pid file but socket exists; clearing socket"
            rm -f "$SOCK_FILE"
        else
            echo "not running"
        fi
        return 0
    fi
    pid="$(cat "$PID_FILE")"
    if kill -0 "$pid" 2>/dev/null; then
        echo "stopping server_35b (pid $pid)…"
        kill "$pid" 2>/dev/null || true
        for _ in $(seq 1 20); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 0.5
        done
        if kill -0 "$pid" 2>/dev/null; then
            echo "  graceful shutdown timed out; sending SIGTERM"
            kill -TERM "$pid" 2>/dev/null || true
            sleep 1
            kill -KILL "$pid" 2>/dev/null || true
        fi
        echo "stopped"
    else
        echo "pid $pid not alive; cleaning"
    fi
    rm -f "$PID_FILE" "$SOCK_FILE"
}

cmd_status() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        if [ -S "$SOCK_FILE" ]; then
            echo "running (pid $(cat "$PID_FILE"), socket $SOCK_FILE)"
        else
            echo "process up but no socket yet (still bootstrapping?)"
        fi
    else
        echo "not running"
        if [ -f "$PID_FILE" ]; then
            rm -f "$PID_FILE"
        fi
    fi
}

cmd_restart() {
    cmd_stop
    sleep 1
    cmd_start
}

case "${1:-}" in
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    status)  cmd_status ;;
    restart) cmd_restart ;;
    *) echo "usage: $0 {start|stop|status|restart}"; exit 2 ;;
esac
