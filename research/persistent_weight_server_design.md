# Persistent Weight-Loaded Inference Server — Design Doc

## Problem statement

Every dev cycle for the Qwen3.6-27B port (`91r`, `perf_baseline`, `demo_qwen36_27b`)
re-opens the device and re-uploads 64 layers of bf8 weights to DRAM. That is
~11 minutes of pure I/O on qb2 before any actual experiment runs. A trivial
edit to `experiments/91f_qwen36_27b_full_ondevice.py` (e.g. tweaking a
`compute_kernel_config`, swapping an op variant, fixing a shape) shouldn't cost
11 minutes per iteration. We need a **long-running server process** on qb2
that holds the device handle and all 64 layers of bf16/bf8 weights resident,
and dispatches test runs against the warm state.

## 1. Architecture recommendation

**Persistent Python process + Unix-domain-socket JSON RPC**, launched under
`nohup` and pinned to qb2. Each entry point (`91r`, `demo`, `perf_baseline`)
becomes a thin client that connects to the socket and submits a job;
weights live in the server's address space for the full session.

This wins over alternatives: a Jupyter kernel ties us to notebook
front-ends and ZMQ machinery we don't otherwise use; a bare REPL is fine for
ad-hoc work but can't be driven from scripts or CI; a file-watcher command
queue introduces filesystem-coupled state and ambiguous "is the request
done?" semantics. A Unix socket is one file (`~/tt-xla/.cache/server.sock`),
supports streaming stdout back to the client, and is trivially `nohup`-able
without TTY allocation.

## 2. Hot-reload semantics

The key invariant: the server holds `device`, `layer_weights` (list of
64 `(layer_type, w_tt)` tuples), `embed_np`, `final_norm_tt`, `lm_head_tt`,
and the tokenizer. These must survive reloads. What gets reloaded is the
**kernel module** — `91f_qwen36_27b_full_ondevice.py` (where
`deltanet_step_ondevice`, `gated_attn_step_ondevice`, `mlp_step_ondevice`
live) and any client entry point.

On a `reload_kernels` command, the server calls `importlib.reload(_91f)`
(or re-runs the `spec_from_file_location` block that `91r` and `demo` use)
and rebinds the step functions in its own namespace. **Closures captured
before reload are stale** — `91r`'s import-time `deltanet_step_ondevice =
_91f.deltanet_step_ondevice` snapshot needs to be redone after reload;
the server must hold the module reference and dereference per-call
(`_91f.deltanet_step_ondevice(...)`) rather than caching the function. The
config dict `cfg` is plain data and survives. Any RoPE freq tables, RMSNorm
EPS constants, MAX_POS — if they live in the kernel module, they're picked
up automatically on reload. State held *inside* the server (KV caches,
SSM/conv states) must be cleared on reload because the new code may have
different shape/dtype expectations.

## 3. State reset between tests

A `reset_state` command rebuilds the per-test transient buffers without
touching `layer_weights`. Specifically: allocate fresh zero-filled
`ssm_state` (`[n_v_heads, k_dim, v_dim] fp32→bf16`), `conv_state` (`[CONV_DIM,
conv_kernel-1] bf16`), and per-attention-layer KV cache pairs
(`[1, n_kv_heads, MAX_POS, head_dim] bf16`) — the exact buffers built by
`fresh_state()` in `demo_qwen36_27b.py`. Position counter resets to 0.
This is cheap (<1 s; just zero tensors at known shapes), runs `gc.collect()`
to free the old buffers, and never re-downloads weights from HF or
re-uploads them to DRAM.

## 4. Commands the server exposes

JSON-over-Unix-socket, one request per connection, newline-delimited
responses for streaming logs.

| Command | Args | Returns |
|---|---|---|
| `status` | — | `{loaded: bool, num_layers, device_id, uptime_sec, last_run}` |
| `reload_kernels` | — | `{ok, reloaded_modules: [...]}` |
| `reset_state` | — | `{ok, dt_sec}` |
| `run_91r` | `{layers: [0,3,7,...], weight_dtype}` | streamed stdout + final result JSON |
| `run_paris` | `{tokens: 40}` | streamed text + cosine sanity |
| `run_perf_baseline` | `{phase, decode_repeats, ...}` | timings + JSON path |
| `shutdown` | — | closes device, exits |

Request wire format: `{"cmd": "...", "args": {...}}`. Streaming logs use
`{"type": "log", "line": "..."}` and the terminal response is
`{"type": "result", "data": {...}}` or `{"type": "error", "msg": "..."}`.

## 5. Failure modes

- **Server crash (Python exception during a run):** wrap the dispatch loop
  in a try/except that catches everything except `SystemExit`, logs the
  traceback to `~/tt-xla/.cache/server.log`, returns the error to the
  client, and keeps the loop running. Weights stay resident; only the
  current request dies.
- **ttnn op assert (C++ abort):** unrecoverable — the process dies, the
  socket closes, the device handle is gone. The client detects EOF and
  reports it. Recovery: relaunch the server (`scripts/serve.sh start`),
  pay the 11-min reload once. We accept this; it's the *baseline* we're
  trying to amortize across many code edits.
- **SSH disconnect:** server runs under `nohup` + `setsid` + redirected
  stdout/stderr; survives the ssh session terminating. Client just
  reconnects to the socket on next ssh.

## 6. File layout

```
experiments/serve/
  server.py            # main loop: socket bind, command dispatch, state holder
  client.py            # thin: connect, send JSON, stream response
  protocol.py          # shared command/response schema dataclasses
  commands/
    run_91r.py         # ports 91r main() to take an injected (device, weights, cfg)
    run_paris.py       # canonical-prompt sanity check
    run_perf.py        # ports perf_baseline main() likewise
    reset_state.py     # the fresh_state() builder factored out
  scripts/
    serve.sh           # start|stop|status wrapper (nohup, pidfile)
```

Existing entry points (`91r`, `demo_qwen36_27b`, `perf_baseline`) stay as
standalone fallbacks that still work without the server; the
`commands/run_*.py` files import the *kernels* (from `91f`) and the
*orchestration logic* of those entry points, factored to take a
pre-built `(device, layer_weights, cfg, embed/lm_head)` instead of
constructing them.

## 7. Phase 1 MVP

**Smallest thing that captures 80% of the value:** server loads all 64
layers once at boot, exposes only `run_91r`, `reset_state`, `reload_kernels`,
`status`, `shutdown`. No streaming logs (just buffer + dump on completion).
No `run_paris`, no `run_perf_baseline` yet. Client is a one-shot CLI:
`python -m experiments.serve.client run_91r --layers 0,3,7,11`.
This alone replaces the most painful workflow — `91r` is the iteration
loop we currently run most.

## 8. Effort estimate

- **Phase 1 (MVP):** 4-6 hours. Server skeleton (1h), `91r` factored to
  take injected state (1.5h), socket protocol + client (1h), `serve.sh`
  start/stop wrapper (0.5h), end-to-end test on qb2 (1-2h including the
  inevitable ttnn handle-lifetime surprises).
- **Phase 2 (full scope):** another 4-6 hours. Add `run_paris`,
  `run_perf_baseline`, streaming logs, proper `reload_kernels` with module
  rebinding, log rotation, graceful op-assert handling that at least
  prints a stack trace before the process dies.

## 9. Top 3 risks

**Risk 1 — ttnn tensor lifetime tied to local scope.** Many ttnn tensors
in our codebase are constructed inside `main()` and freed when it returns.
A server keeps `main()` running forever; if any tensor (especially
intermediate activations during a run) lives in module-globals it could
leak across runs. Mitigation: every command handler is a function-local
scope, returns only Python data (cosines, timings, text), and explicitly
calls `gc.collect()` after each run.

**Risk 2 — kernel-reload staleness.** `importlib.reload` rebinds the
module's *globals* but doesn't touch already-captured references. If the
server caches `from _91f import deltanet_step_ondevice` (like `91r` does
at import time), reload won't pick up edits. Mitigation: never cache; always
dereference via `_91f.<fn>` at call time. Verify with a smoke test that
edits a `print` into `deltanet_step_ondevice` and checks the log shows it.

**Risk 3 — ttnn op-assert kills everything.** Most of our debug cycles
trigger exactly the kind of shape/dtype mismatches that abort the process.
The 11-min reload returns the moment we crash. Mitigation: this is
unavoidable for unrecoverable C++ aborts, but we minimize the blast radius
by validating shapes/dtypes in Python preflight before dispatching the
ttnn call. Optional Phase 3: a subprocess worker pool so an op-assert
only kills the worker, not the weight-holding parent — but that
re-introduces IPC of device tensors, which is non-trivial; defer until
the simpler design proves insufficient.
