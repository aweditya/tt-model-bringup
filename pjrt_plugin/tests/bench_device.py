"""Performance benchmarks for the TT PJRT engine.

Phase 5 Step 5. Measures four surfaces:
    1. Raw ttnn  — bare ops, both tensors already on device.
    2. Engine eager  — same ops via engine._execute_op_device (tensors on device).
    3. Engine end-to-end  — engine.execute_stablehlo(bytecode, numpy_inputs).
    4. PJRT via jax.jit  — the C++-shim path.

Run on qb1:
    cd ~/tt-xla
    TT_PJRT_USE_DEVICE=1 .venv/bin/python pjrt_plugin/tests/bench_device.py

This script is permanent — re-run after Step 6 (trace capture) and again
after Step 7 (op fusion) to track speedups. Results append to
research/pjrt_phase5_benchmarks.md.
"""

import argparse
import os
import sys
import time
import subprocess
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PJRT_DIR = REPO_ROOT / "pjrt_plugin"
sys.path.insert(0, str(PJRT_DIR))

os.environ.setdefault("TT_PJRT_USE_DEVICE", "1")

# Import the engine the same way the C++ plugin / JAX would, so we share
# the SAME module instance (and the SAME `_device` global). Without this
# the bench would open the Blackhole twice and crash on the second.
from jax_plugins.tt import engine

if engine._USE_DEVICE:
    import ttnn  # exported by engine after dual-mode init
    import torch
else:
    raise SystemExit(
        "TT_PJRT_USE_DEVICE=1 required; engine.py says ttnn isn't loaded.")


# -----------------------------------------------------------
# Timing primitives
# -----------------------------------------------------------

WARMUP = 10
ITERS = 200


def _sync():
    """Wait for the device queue to drain. Required for true op timing."""
    ttnn.synchronize_device(engine._get_device())


def time_loop(fn, warmup=WARMUP, iters=ITERS, sync_each=True):
    """Run fn() warmup+iters times. Return mean and p99 in microseconds.

    sync_each=True means we sync after every iter — that gives true
    completion-time per call. sync_each=False is dispatch-only timing
    (useful to highlight queue throughput).
    """
    for _ in range(warmup):
        fn()
    if sync_each:
        _sync()

    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if sync_each:
            _sync()
        samples.append(time.perf_counter() - t0)

    samples_us = np.array(samples) * 1e6
    return float(samples_us.mean()), float(np.percentile(samples_us, 99))


# -----------------------------------------------------------
# Surface 1 — raw ttnn on tensors already on device
# -----------------------------------------------------------

def bench_raw_ttnn():
    results = {}

    a_1x32 = engine._to_device(np.random.randn(1, 32).astype(np.float32))
    b_1x32 = engine._to_device(np.random.randn(1, 32).astype(np.float32))

    a_32x32 = engine._to_device(np.random.randn(32, 32).astype(np.float32))
    b_32x32 = engine._to_device(np.random.randn(32, 32).astype(np.float32))

    a_64x64 = engine._to_device(np.random.randn(64, 64).astype(np.float32) * 0.1)
    b_64x64 = engine._to_device(np.random.randn(64, 64).astype(np.float32) * 0.1)

    a_256 = engine._to_device(np.random.randn(256, 256).astype(np.float32) * 0.05)
    b_256 = engine._to_device(np.random.randn(256, 256).astype(np.float32) * 0.05)

    mean, p99 = time_loop(lambda: ttnn.add(a_1x32, b_1x32))
    results["ttnn.add 1x32"] = (mean, p99)

    mean, p99 = time_loop(lambda: ttnn.add(a_32x32, b_32x32))
    results["ttnn.add 32x32"] = (mean, p99)

    mean, p99 = time_loop(lambda: ttnn.exp(a_1x32))
    results["ttnn.exp 1x32"] = (mean, p99)

    mean, p99 = time_loop(lambda: ttnn.matmul(a_64x64, b_64x64))
    results["ttnn.matmul 64x64"] = (mean, p99)

    mean, p99 = time_loop(lambda: ttnn.matmul(a_256, b_256))
    results["ttnn.matmul 256x256"] = (mean, p99)

    return results


# -----------------------------------------------------------
# Surface 2 — engine._execute_op_device with tensors on device
# -----------------------------------------------------------

def bench_engine_eager():
    results = {}

    a_1x32 = engine._to_device(np.random.randn(1, 32).astype(np.float32))
    b_1x32 = engine._to_device(np.random.randn(1, 32).astype(np.float32))
    add_op = {"name": "0", "op": "add", "operands": ["a", "b"],
              "result_type": "tensor<1x32xf32>"}
    vals = {"a": a_1x32, "b": b_1x32}
    mean, p99 = time_loop(lambda: engine._execute_op_device(add_op, vals))
    results["engine.add 1x32"] = (mean, p99)

    exp_op = {"name": "0", "op": "exp", "operands": ["a"],
              "result_type": "tensor<1x32xf32>"}
    vals = {"a": a_1x32}
    mean, p99 = time_loop(lambda: engine._execute_op_device(exp_op, vals))
    results["engine.exp 1x32"] = (mean, p99)

    a_64 = engine._to_device(np.random.randn(64, 64).astype(np.float32) * 0.1)
    b_64 = engine._to_device(np.random.randn(64, 64).astype(np.float32) * 0.1)
    mm_op = {"name": "0", "op": "dot_general", "operands": ["a", "b"],
             "lhs_contracting": [1], "rhs_contracting": [0],
             "lhs_batching": [], "rhs_batching": [],
             "result_type": "tensor<64x64xf32>"}
    vals = {"a": a_64, "b": b_64}
    mean, p99 = time_loop(lambda: engine._execute_op_device(mm_op, vals))
    results["engine.matmul 64x64"] = (mean, p99)

    a_256 = engine._to_device(np.random.randn(256, 256).astype(np.float32) * 0.05)
    b_256 = engine._to_device(np.random.randn(256, 256).astype(np.float32) * 0.05)
    mm_op_256 = dict(mm_op)
    mm_op_256["result_type"] = "tensor<256x256xf32>"
    vals = {"a": a_256, "b": b_256}
    mean, p99 = time_loop(lambda: engine._execute_op_device(mm_op_256, vals))
    results["engine.matmul 256x256"] = (mean, p99)

    return results


# -----------------------------------------------------------
# Surface 3 — engine.execute_stablehlo end-to-end
# -----------------------------------------------------------

def _serialize_bytecode(fn, *example_args):
    """Build the VHLO portable artifact that JAX hands to a PJRT plugin.

    Identical to the format C++ Compile receives. The engine's
    bytecode_to_text only correctly parses portable artifacts (registers
    the right dialects); raw MLIR bytecode would miss the func dialect.
    """
    import jax
    from jaxlib.mlir._mlir_libs._stablehlo import (
        serialize_portable_artifact, get_current_version,
    )
    cpu_dev = jax.devices("cpu")[0]
    with jax.default_device(cpu_dev):
        lowered = jax.jit(fn).lower(*example_args)
        module = lowered.compiler_ir(dialect="stablehlo")
    version = get_current_version()
    return serialize_portable_artifact(module, version)


_E2E_PROGRAMS = None  # built lazily and shared across eager + traced surfaces


def _build_e2e_programs():
    """Return list of (label, bytecode, [inputs]) tuples used by surfaces 3+5."""
    global _E2E_PROGRAMS
    if _E2E_PROGRAMS is not None:
        return _E2E_PROGRAMS
    import jax.numpy as jnp

    x_4 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    a64 = np.random.randn(64, 64).astype(np.float32) * 0.1
    b64 = np.random.randn(64, 64).astype(np.float32) * 0.1
    a64s = np.random.randn(2, 64).astype(np.float32) * 0.1
    w64 = np.random.randn(64, 32).astype(np.float32) * 0.1
    b32 = np.random.randn(32).astype(np.float32) * 0.1
    x_64 = np.random.randn(1, 64).astype(np.float32)

    def softmax(x):
        m = jnp.max(x, axis=-1, keepdims=True)
        e = jnp.exp(x - m)
        return e / jnp.sum(e, axis=-1, keepdims=True)

    progs = [
        ("x + 1 (1-op)", _serialize_bytecode(lambda x: x + 1.0, x_4), [x_4]),
        ("exp(x) (1-op)", _serialize_bytecode(lambda x: jnp.exp(x), x_4), [x_4]),
        ("a @ b 64x64", _serialize_bytecode(lambda a, b: a @ b, a64, b64),
         [a64, b64]),
        ("linear (a@w+b)",
         _serialize_bytecode(lambda a, w, b: a @ w + b, a64s, w64, b32),
         [a64s, w64, b32]),
        ("softmax (1x64)", _serialize_bytecode(softmax, x_64), [x_64]),
    ]
    _E2E_PROGRAMS = progs
    return progs


def bench_engine_end_to_end():
    """Surface 3 — eager-only (trace disabled)."""
    results = {}
    # Disable trace cache for this surface so we get true eager numbers.
    saved = engine._NO_TRACE
    engine._NO_TRACE = True
    try:
        for label, bc, inputs in _build_e2e_programs():
            fn = lambda bc=bc, inputs=inputs: engine.execute_stablehlo(bc, inputs)
            mean, p99 = time_loop(fn, sync_each=False)
            results[f"e2e: {label}"] = (mean, p99)
    finally:
        engine._NO_TRACE = saved
    return results


def bench_engine_traced():
    """Surface 5 — trace capture enabled (cache hit path)."""
    results = {}
    engine._NO_TRACE = False
    for label, bc, inputs in _build_e2e_programs():
        # First call captures the trace; subsequent calls replay.
        engine.execute_stablehlo(bc, inputs)  # capture
        fn = lambda bc=bc, inputs=inputs: engine.execute_stablehlo(bc, inputs)
        mean, p99 = time_loop(fn, sync_each=False)
        cache_entry = engine._trace_cache.get(hash(bc), {})
        suffix = "" if not cache_entry.get('failed') else " [no-trace]"
        results[f"traced: {label}{suffix}"] = (mean, p99)
    return results


# -----------------------------------------------------------
# Surface: parse-only (bytecode→ops, no execution)
# -----------------------------------------------------------

def bench_parse_only():
    results = {}
    import jax.numpy as jnp

    x_4 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    bc = _serialize_bytecode(lambda x: x + 1.0, x_4)

    def parse_once():
        text = engine.bytecode_to_text(bc)
        engine.parse_stablehlo(text)

    mean, p99 = time_loop(parse_once, sync_each=False)
    results["parse: x + 1"] = (mean, p99)

    # Bigger program
    x_64 = np.random.randn(1, 64).astype(np.float32)
    def softmax(x):
        m = jnp.max(x, axis=-1, keepdims=True)
        e = jnp.exp(x - m)
        return e / jnp.sum(e, axis=-1, keepdims=True)
    bc = _serialize_bytecode(softmax, x_64)

    def parse_softmax():
        text = engine.bytecode_to_text(bc)
        engine.parse_stablehlo(text)

    mean, p99 = time_loop(parse_softmax, sync_each=False)
    results["parse: softmax"] = (mean, p99)

    return results


# -----------------------------------------------------------
# Surface 4 — jax.jit through the PJRT plugin
# -----------------------------------------------------------

def bench_jax_jit():
    results = {}
    try:
        import jax
        import jax.numpy as jnp
        from jax_plugins.tt import initialize as _init_plugin
        try:
            _init_plugin()
        except Exception:
            pass
        tt_devs = jax.devices("tt")
        if not tt_devs:
            return {"_skip": (0, 0)}
        tt = tt_devs[0]
    except Exception as e:
        return {"_skip_reason": str(e)}

    # x + 1
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    f1 = jax.jit(lambda x: x + 1.0)
    x_dev = jax.device_put(x, tt)
    for _ in range(WARMUP):
        jax.device_get(f1(x_dev))
    mean, p99 = time_loop(lambda: jax.device_get(f1(x_dev)),
                          sync_each=False, warmup=0)
    results["jit: x + 1"] = (mean, p99)

    # exp(x)
    f2 = jax.jit(lambda x: jnp.exp(x))
    for _ in range(WARMUP):
        jax.device_get(f2(x_dev))
    mean, p99 = time_loop(lambda: jax.device_get(f2(x_dev)),
                          sync_each=False, warmup=0)
    results["jit: exp(x)"] = (mean, p99)

    # matmul
    a = np.random.randn(64, 64).astype(np.float32) * 0.1
    b = np.random.randn(64, 64).astype(np.float32) * 0.1
    f3 = jax.jit(lambda a, b: a @ b)
    a_dev = jax.device_put(a, tt)
    b_dev = jax.device_put(b, tt)
    for _ in range(WARMUP):
        jax.device_get(f3(a_dev, b_dev))
    mean, p99 = time_loop(lambda: jax.device_get(f3(a_dev, b_dev)),
                          sync_each=False, warmup=0)
    results["jit: a @ b 64x64"] = (mean, p99)

    return results


# -----------------------------------------------------------
# Output
# -----------------------------------------------------------

def format_table(title, results):
    lines = [f"### {title}", "",
             "| op | mean (us) | p99 (us) |", "|---|---:|---:|"]
    for k, v in results.items():
        if k.startswith("_skip"):
            lines.append(f"| _skipped: {v} | - | - |")
            continue
        mean, p99 = v
        lines.append(f"| {k} | {mean:.1f} | {p99:.1f} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "research" / "pjrt_phase5_benchmarks.md")
    ap.add_argument("--label", default="", help="Annotate this run (e.g. 'baseline-eager').")
    args = ap.parse_args()

    print("Opening device + warming kernel cache...")
    _ = engine._get_device()
    # Warm tensor creation cache too
    _ = engine._to_device(np.zeros((32, 32), dtype=np.float32))
    _sync()

    sections = []

    surfaces = [
        ("Surface 1 — Raw ttnn (tensors on device)", bench_raw_ttnn),
        ("Surface 2 — Engine eager (_execute_op_device)", bench_engine_eager),
        ("Surface (parse) — bytecode_to_text + parse_stablehlo", bench_parse_only),
        ("Surface 3 — Engine end-to-end (eager, no trace)", bench_engine_end_to_end),
        ("Surface 5 — Engine traced (begin/end_trace_capture)", bench_engine_traced),
        ("Surface 4 — jax.jit (full PJRT pipeline)", bench_jax_jit),
    ]
    for i, (title, fn) in enumerate(surfaces, 1):
        print(f"\n[{i}/{len(surfaces)}] {title}")
        try:
            sections.append((title, fn()))
        except Exception as e:
            import traceback
            print(f"    !! {title} failed: {e}")
            traceback.print_exc()
            sections.append((title, {"_error": (0, 0), "_msg: " + str(e)[:80]: (0, 0)}))

    # Render
    git_sha = ""
    try:
        git_sha = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True).strip()
    except Exception:
        pass
    header = (f"\n## Run {time.strftime('%Y-%m-%d %H:%M:%S')} "
              f"(sha={git_sha}{', '+args.label if args.label else ''})\n")

    body = header + "\n" + "\n".join(format_table(t, r) for t, r in sections)
    print(body)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not args.out.exists():
        args.out.write_text(
            "# Phase 5 Benchmark Results\n\n"
            "Run with `bench_device.py`. Each section is one run.\n")
    with args.out.open("a") as f:
        f.write(body)
    print(f"\nAppended to {args.out}")


if __name__ == "__main__":
    main()
