#!/usr/bin/env bash
# serve_cb.sh — start | stop | status | restart for the CB OpenAI HTTP server.
# Parallel to serve_tp.sh (which serves the single-seq Unix-socket prod path);
# this targets `experiments.serve.cb_api:app` under uvicorn, which boots a
# CBEngine(sampling=True) over the (1, 4) mesh device in its lifespan.
#
# Run from the tt-xla repo root:  bash experiments/serve/scripts/serve_cb.sh start
#
# Environment knobs (with defaults):
#   TT_CB_PORT=8000          HTTP listen port
#   TT_CB_HOST=0.0.0.0       HTTP listen address
#   TT_CB_SLOTS=4            CB scheduler slots
#   TT_CB_MAX_NEW=1024       per-request max_new_tokens cap
#   TT_CB_MAX_INFLIGHT=64    queue+active in-flight cap (over-cap → HTTP 429)
#   TT_CB_TOPK_K             (unset)=full-vocab logits trace (best at low B);
#                            set to e.g. 128 to enable on-device top-k (best at B≥16)
set -u

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/tt-xla}"
CACHE_DIR="$PROJECT_ROOT/.cache"
PID_FILE="$CACHE_DIR/server_cb.pid"
LOG_FILE="$CACHE_DIR/server_cb.log"
VENV_PY="${VENV_PY:-$PROJECT_ROOT/.venv/bin/python}"
PORT="${TT_CB_PORT:-8000}"
HOST_ADDR="${TT_CB_HOST:-0.0.0.0}"

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
        echo "server_cb already running (pid $(cat "$PID_FILE"))"
        exit 0
    fi
    rm -f "$PID_FILE"
    cd "$PROJECT_ROOT" || { echo "cannot cd $PROJECT_ROOT"; exit 1; }
    [ -x "$VENV_PY" ] || { echo "venv python not found at $VENV_PY"; exit 1; }
    echo "starting server_cb on $HOST_ADDR:$PORT (logs: $LOG_FILE)…"
    : > "$LOG_FILE"
    TT_METAL_HOME="${TT_METAL_HOME:-$HOME/tenstorrent/tt-metal}"
    TT_BUILD_DIR="${TT_BUILD_DIR:-$TT_METAL_HOME/build_Release}"
    HF_HOME="${HF_HOME:-$CACHE_DIR/hf}" \
    TT_METAL_HOME="$TT_METAL_HOME" \
    TT_BUILD_DIR="$TT_BUILD_DIR" \
    ARCH_NAME="${ARCH_NAME:-blackhole}" \
    PYTHONPATH="$TT_METAL_HOME/ttnn:${PYTHONPATH:-}" \
    LD_LIBRARY_PATH="$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib:${LD_LIBRARY_PATH:-}" \
    PYTHONUNBUFFERED=1 \
    TT_CB_SLOTS="${TT_CB_SLOTS:-4}" \
    TT_CB_MAX_NEW="${TT_CB_MAX_NEW:-1024}" \
    TT_CB_MAX_INFLIGHT="${TT_CB_MAX_INFLIGHT:-64}" \
    TT_CB_TOPK_K="${TT_CB_TOPK_K:-0}" \
    nohup setsid "$VENV_PY" -m uvicorn experiments.serve.cb_api:app \
        --host "$HOST_ADDR" --port "$PORT" --lifespan on \
        >> "$LOG_FILE" 2>&1 < /dev/null &
    local pid=$!
    echo "$pid" > "$PID_FILE"
    sleep 1
    if is_running; then
        echo "launched (pid $pid) — bootstrap takes ~6 min for sharded weight load"
        echo "  /health stays 503 until the engine is up"
        echo "  tail -f $LOG_FILE"
    else
        echo "launch failed — see $LOG_FILE"; tail -n 40 "$LOG_FILE" 2>/dev/null || true
        rm -f "$PID_FILE"; exit 1
    fi
}

cmd_stop() {
    if ! is_running; then
        echo "server_cb not running"
        rm -f "$PID_FILE"
        return 0
    fi
    local pid; pid=$(cat "$PID_FILE")
    echo "stopping server_cb (pid $pid)…"
    # uvicorn handles SIGTERM as graceful shutdown: stops accepting connections,
    # drains in-flight requests, then runs the lifespan teardown (engine.stop()
    # → mesh release). Hard-killing mid-mesh-init can wedge the fabric; if that
    # happens, recover with `tt-smi -r 0,1,2,3`.
    kill "$pid" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
        echo "  graceful shutdown timed out; SIGKILL (fabric may need tt-smi -r 0,1,2,3)"
        kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
    echo "stopped"
}

cmd_status() {
    if is_running; then
        local pid; pid=$(cat "$PID_FILE")
        echo "running (pid $pid, http://$HOST_ADDR:$PORT)"
        # /health returns 200 only after the engine is bootstrapped + ready.
        if command -v curl >/dev/null 2>&1; then
            local code
            code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 "http://127.0.0.1:$PORT/health" 2>/dev/null || echo "?")
            echo "  /health -> HTTP $code"
        fi
    else
        echo "not running"
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
