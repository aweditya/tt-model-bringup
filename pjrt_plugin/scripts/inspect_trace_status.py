"""Inspect which programs hit the trace cache vs fall back to eager.

Step 7 debug helper. Runs a few composite programs (softmax, layernorm,
rmsnorm, attention) through engine.execute_stablehlo, then prints the
trace cache state for each: did the trace capture succeed, or did the
program fall back to parse-cached eager?

Run on qb1:
    cd ~/tt-xla
    TT_PJRT_USE_DEVICE=1 .venv/bin/python pjrt_plugin/scripts/inspect_trace_status.py
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "pjrt_plugin"))

os.environ.setdefault("TT_PJRT_USE_DEVICE", "1")

import numpy as np
from jax_plugins.tt import engine


def serialize(fn, *example_args):
    import jax
    from jaxlib.mlir._mlir_libs._stablehlo import (
        serialize_portable_artifact, get_current_version,
    )
    cpu_dev = jax.devices("cpu")[0]
    with jax.default_device(cpu_dev):
        lowered = jax.jit(fn).lower(*example_args)
        module = lowered.compiler_ir(dialect="stablehlo")
    return serialize_portable_artifact(module, get_current_version())


def run_and_inspect(label, fn, *args):
    bc = serialize(fn, *args)
    # Run twice — first call captures, second replays (or falls back).
    np_args = [np.asarray(a) for a in args]
    r1 = engine.execute_stablehlo(bc, np_args)
    r2 = engine.execute_stablehlo(bc, np_args)

    key = hash(bc)
    entry = engine._trace_cache.get(key, {})
    if entry.get('failed'):
        status = f"FAILED ({entry.get('error', '?')[:60]})"
    elif 'trace_id' in entry:
        status = f"OK (trace_id={entry['trace_id']})"
    else:
        status = f"unknown: {list(entry.keys())}"

    # Correctness check: r1 should match r2
    ok = True
    for a, b in zip(r1, r2):
        if not np.allclose(np.asarray(a), np.asarray(b), atol=1e-2, rtol=1e-2):
            ok = False
            break

    print(f"  {label:30s} -> {status}  | match: {ok}")
    return entry


def main():
    print("Opening device...")
    _ = engine._get_device()

    import jax
    import jax.numpy as jnp

    print("\n--- Programs ---")

    x_64 = np.random.randn(1, 64).astype(np.float32)
    x_2x64 = np.random.randn(2, 64).astype(np.float32)
    g_64 = np.ones(64, dtype=np.float32)
    b_64 = np.zeros(64, dtype=np.float32)

    run_and_inspect("x + 1",
                    lambda x: x + 1.0, np.array([1., 2., 3., 4.], dtype=np.float32))

    a64 = np.random.randn(64, 64).astype(np.float32) * 0.1
    b64 = np.random.randn(64, 64).astype(np.float32) * 0.1
    run_and_inspect("matmul 64x64",
                    lambda a, b: a @ b, a64, b64)

    a64s = np.random.randn(2, 64).astype(np.float32) * 0.1
    w64 = np.random.randn(64, 32).astype(np.float32) * 0.1
    b32 = np.random.randn(32).astype(np.float32) * 0.1
    run_and_inspect("linear (a@w+b)",
                    lambda a, w, b: a @ w + b, a64s, w64, b32)

    run_and_inspect("softmax (2x64)",
                    lambda x: jax.nn.softmax(x, axis=-1), x_2x64)

    def layer_norm(x, g, b):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
        return g * (x - mean) / jnp.sqrt(var + 1e-5) + b
    run_and_inspect("layer_norm (2x64)",
                    layer_norm, x_2x64, g_64, b_64)

    def rms_norm(x, g):
        ms = jnp.mean(x ** 2, axis=-1, keepdims=True)
        return g * x / jnp.sqrt(ms + 1e-6)
    run_and_inspect("rms_norm (2x64)",
                    rms_norm, x_2x64, g_64)

    def attention(x, wq, wk, wv, wo):
        q = x @ wq; k = x @ wk; v = x @ wv
        d = jnp.float32(q.shape[-1])
        scores = jax.nn.softmax(q @ k.T / jnp.sqrt(d), axis=-1)
        return (scores @ v) @ wo
    D = 32
    x = np.random.randn(8, D).astype(np.float32) * 0.1
    wq = np.random.randn(D, D).astype(np.float32) * 0.1
    wk = np.random.randn(D, D).astype(np.float32) * 0.1
    wv = np.random.randn(D, D).astype(np.float32) * 0.1
    wo = np.random.randn(D, D).astype(np.float32) * 0.1
    run_and_inspect("attention (8x32)", attention, x, wq, wk, wv, wo)


if __name__ == "__main__":
    main()
