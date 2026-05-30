#!/usr/bin/env bash
# serve.sh — start | stop | status | restart for the persistent weight server.
# Run from the tt-xla repo root: bash experiments/serve/scripts/serve.sh start
set -u

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/tt-xla}"
CACHE_DIR="$PROJECT_ROOT/.cache"
PID_FILE="$CACHE_DIR/server.pid"
SOCK_FILE="$CACHE_DIR/server.sock"
LOG_FILE="$CACHE_DIR/server.log"
VENV_PY="${VENV_PY:-$PROJECT_ROOT/.venv/bin/python}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$CACHE_DIR"

is_running() {
    [ -f "$PID_FILE" ] || return 1
    local pid
    pid=$(cat "$PID_FILE" 2>/dev/null) || return 1
    [ -n "$pid" ] || return 1
    kill -0 "$pid" 2>/dev/null
}

cmd_start() {
    if is_running; then
        echo "server already running (pid $(cat "$PID_FILE"))"
        exit 0
    fi
    rm -f "$SOCK_FILE" "$PID_FILE"
    cd "$PROJECT_ROOT" || { echo "cannot cd $PROJECT_ROOT"; exit 1; }
    if [ ! -x "$VENV_PY" ]; then
        echo "venv python not found at $VENV_PY"; exit 1
    fi
    echo "starting server (logs: $LOG_FILE)…"
    : > "$LOG_FILE"
    # setsid detaches from controlling terminal; nohup keeps it alive across
    # the parent shell exiting. PYTHONUNBUFFERED for prompt log flushing.
    # Full TT env block — without these, `import ttnn` either fails or picks up
    # a stale wheel silently. Mirrors scripts/run_remote.sh:22-28.
    TT_METAL_HOME="${TT_METAL_HOME:-$HOME/tenstorrent/tt-metal}"
    TT_BUILD_DIR="${TT_BUILD_DIR:-$TT_METAL_HOME/build_Release}"
    HF_HOME="${HF_HOME:-$CACHE_DIR/hf}" \
    TT_METAL_HOME="$TT_METAL_HOME" \
    TT_BUILD_DIR="$TT_BUILD_DIR" \
    ARCH_NAME="${ARCH_NAME:-blackhole}" \
    PYTHONPATH="$TT_METAL_HOME/ttnn:${PYTHONPATH:-}" \
    LD_LIBRARY_PATH="$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib:${LD_LIBRARY_PATH:-}" \
    PYTHONUNBUFFERED=1 \
    nohup setsid "$VENV_PY" -m experiments.serve.server $EXTRA_ARGS \
        >> "$LOG_FILE" 2>&1 < /dev/null &
    local pid=$!
    # pidfile is written by the server itself; record the launcher's view too
    # in case bootstrap dies before it gets that far.
    echo "$pid" > "$PID_FILE.launch"
    sleep 1
    if is_running || kill -0 "$pid" 2>/dev/null; then
        echo "launched (pid $pid)  — bootstrap may take ~11 min for full weight load"
        echo "  tail -f $LOG_FILE"
    else
        echo "launch failed — see $LOG_FILE"
        tail -n 40 "$LOG_FILE" 2>/dev/null || true
        exit 1
    fi
}

cmd_stop() {
    if ! is_running; then
        echo "server not running"
        rm -f "$PID_FILE" "$SOCK_FILE" "$PID_FILE.launch"
        return 0
    fi
    local pid; pid=$(cat "$PID_FILE")
    echo "stopping server (pid $pid)…"
    # Try graceful shutdown via socket first.
    if [ -S "$SOCK_FILE" ] && [ -x "$VENV_PY" ]; then
        (cd "$PROJECT_ROOT" && timeout 10 "$VENV_PY" -m experiments.serve.client shutdown \
            > /dev/null 2>&1) || true
        for _ in 1 2 3 4 5; do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
    fi
    if kill -0 "$pid" 2>/dev/null; then
        echo "  graceful shutdown timed out; sending SIGTERM"
        kill "$pid" 2>/dev/null || true
        for _ in 1 2 3 4 5; do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
    fi
    if kill -0 "$pid" 2>/dev/null; then
        echo "  still alive; SIGKILL"
        kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE" "$SOCK_FILE" "$PID_FILE.launch"
    echo "stopped"
}

cmd_status() {
    if is_running; then
        local pid; pid=$(cat "$PID_FILE")
        echo "running (pid $pid, socket $SOCK_FILE)"
        if [ -x "$VENV_PY" ] && [ -S "$SOCK_FILE" ]; then
            (cd "$PROJECT_ROOT" && "$VENV_PY" -m experiments.serve.client status) || true
        fi
    else
        echo "not running"
        if [ -f "$PID_FILE.launch" ]; then
            echo "  launcher recorded pid $(cat "$PID_FILE.launch") but it's gone — see $LOG_FILE"
            tail -n 20 "$LOG_FILE" 2>/dev/null || true
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
