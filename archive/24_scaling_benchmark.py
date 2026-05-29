"""
Experiment 24: Scaling benchmark — how does Blackhole scale with model size?

Tests the traced transformer at different sequence lengths and hidden dims
to understand the throughput/latency curve. This tells us where Blackhole
shines (small/medium models with low latency) vs where it's limited.

Also benchmarks raw matmul throughput at different sizes to estimate TFLOPS.
"""

import sys, os
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import jax
import jax.numpy as jnp
from jax import make_jaxpr
import ttnn
import time

from tt_jax.interpret import Interpreter
from tt_jax import tensors


def transformer_block(x, w_q, w_k, w_v, w_o, w1, w2, g1, b1, g2, b2):
    q = jnp.dot(x, w_q); k = jnp.dot(x, w_k); v = jnp.dot(x, w_v)
    scores = jnp.dot(q, k.T) / jnp.sqrt(jnp.array(float(x.shape[-1])))
    attn = jax.nn.softmax(scores, axis=-1)
    context = jnp.dot(attn, v)
    h = x + jnp.dot(context, w_o)
    m1 = jnp.mean(h, axis=-1, keepdims=True)
    v1 = jnp.mean((h - m1) ** 2, axis=-1, keepdims=True)
    h = g1 * (h - m1) / jnp.sqrt(v1 + 1e-5) + b1
    ff = jax.nn.relu(jnp.dot(h, w1))
    ff = jnp.dot(ff, w2)
    h2 = h + ff
    m2 = jnp.mean(h2, axis=-1, keepdims=True)
    v2 = jnp.mean((h2 - m2) ** 2, axis=-1, keepdims=True)
    out = g2 * (h2 - m2) / jnp.sqrt(v2 + 1e-5) + b2
    return out


def benchmark_config(seq_len, d_model, d_ff, device):
    """Benchmark a specific transformer configuration."""
    rng = np.random.RandomState(42)
    s = 0.01
    args = [
        rng.randn(seq_len, d_model).astype(np.float32) * s,
        rng.randn(d_model, d_model).astype(np.float32) * s,
        rng.randn(d_model, d_model).astype(np.float32) * s,
        rng.randn(d_model, d_model).astype(np.float32) * s,
        rng.randn(d_model, d_model).astype(np.float32) * s,
        rng.randn(d_model, d_ff).astype(np.float32) * s,
        rng.randn(d_ff, d_model).astype(np.float32) * s,
        np.ones(d_model, dtype=np.float32),
        np.zeros(d_model, dtype=np.float32),
        np.ones(d_model, dtype=np.float32),
        np.zeros(d_model, dtype=np.float32),
    ]
    jax_args = [jnp.array(a) for a in args]

    try:
        jaxpr = make_jaxpr(transformer_block)(*jax_args)
    except Exception as e:
        return None, str(e)

    # Pre-materialize literals
    literal_cache = {}
    for eqn in jaxpr.jaxpr.eqns:
        for v in eqn.invars:
            if tensors.is_literal(v):
                fval = float(v.val)
                if fval not in literal_cache:
                    literal_cache[fval] = tensors.to_device(v.val, device)

    input_tensors = [tensors.to_device(a, device) for a in args]

    # Warmup
    interp = Interpreter(device, literal_cache=literal_cache)
    for var, t in zip(jaxpr.jaxpr.invars, input_tensors):
        interp.env[var] = t
    for eqn in jaxpr.jaxpr.eqns:
        interp._exec(eqn)

    # Trace capture
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    for eqn in jaxpr.jaxpr.eqns:
        interp._exec(eqn)
    ttnn.end_trace_capture(device, tid, cq_id=0)

    # Correctness check
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    out_var = jaxpr.jaxpr.outvars[0]
    result = tensors.from_device(interp.env[out_var], out_var.aval.shape)
    ref = np.array(transformer_block(*jax_args))
    max_err = np.abs(result - ref).max()

    # Warmup benchmark
    for _ in range(10):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)

    # Benchmark
    N = 200
    t0 = time.perf_counter()
    for _ in range(N):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    t_avg = (time.perf_counter() - t0) / N

    # Compute FLOPS (rough: 7 matmuls dominate)
    # 4 attention matmuls: seq×d @ d×d = 2*seq*d*d each
    # attention scores: seq×d @ d×seq = 2*seq*seq*d
    # context: seq×seq @ seq×d = 2*seq*seq*d
    # 2 FFN matmuls: seq×d @ d×ff = 2*seq*d*ff each
    flops = 4 * (2 * seq_len * d_model * d_model) + \
            2 * (2 * seq_len * seq_len * d_model) + \
            2 * (2 * seq_len * d_model * d_ff)
    tflops = flops / t_avg / 1e12

    ttnn.release_trace(device, tid)

    return {
        'seq_len': seq_len,
        'd_model': d_model,
        'd_ff': d_ff,
        'latency_ms': t_avg * 1000,
        'fwd_per_sec': 1 / t_avg,
        'max_err': max_err,
        'tflops': tflops,
        'n_eqns': len(jaxpr.jaxpr.eqns),
    }, None


def benchmark_matmul(m, n, k, device):
    """Benchmark raw matmul throughput."""
    a = tensors.to_device(np.random.randn(m, k).astype(np.float32) * 0.01, device)
    b = tensors.to_device(np.random.randn(k, n).astype(np.float32) * 0.01, device)

    # Warmup
    c = ttnn.matmul(a, b)

    # Trace
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    c = ttnn.matmul(a, b)
    ttnn.end_trace_capture(device, tid, cq_id=0)

    for _ in range(10):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)

    N = 500
    t0 = time.perf_counter()
    for _ in range(N):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    t_avg = (time.perf_counter() - t0) / N

    flops = 2 * m * n * k
    tflops = flops / t_avg / 1e12

    ttnn.release_trace(device, tid)
    return t_avg * 1000, tflops


# ── Main ──────────────────────────────────────────────────────
device = ttnn.open_device(device_id=0)

# Part 1: Raw matmul throughput at different sizes
print("=" * 60)
print("Part 1: Raw matmul throughput (traced)")
print("=" * 60)
print(f"{'Size':>20s}  {'Latency':>10s}  {'TFLOPS':>8s}")
print("-" * 42)

matmul_configs = [
    (32, 32, 32),
    (64, 64, 64),
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
]

for m, n, k in matmul_configs:
    try:
        lat, tflops = benchmark_matmul(m, n, k, device)
        print(f"{m}x{k} @ {k}x{n}  {lat:>8.3f} ms  {tflops:>7.3f} TF")
    except Exception as e:
        print(f"{m}x{k} @ {k}x{n}  FAILED: {str(e)[:60]}")

# Part 2: Transformer at different scales
print("\n" + "=" * 60)
print("Part 2: Transformer encoder (traced)")
print("=" * 60)
print(f"{'Config':>25s}  {'Latency':>10s}  {'fwd/sec':>10s}  {'TFLOPS':>8s}  {'Error':>8s}")
print("-" * 70)

configs = [
    (32, 64, 256),     # Tiny
    (32, 128, 512),    # Small
    (64, 128, 512),    # Medium-small
    (64, 256, 1024),   # Medium
    (128, 256, 1024),  # Medium-large
    (128, 512, 2048),  # Large
    (256, 512, 2048),  # XL
]

for seq, d, ff in configs:
    try:
        result, err = benchmark_config(seq, d, ff, device)
        if err:
            print(f"seq={seq:>3d} d={d:>3d} ff={ff:>4d}  FAILED: {err[:50]}")
        else:
            print(f"seq={seq:>3d} d={d:>3d} ff={ff:>4d}  "
                  f"{result['latency_ms']:>8.3f} ms  "
                  f"{result['fwd_per_sec']:>8.0f}/s  "
                  f"{result['tflops']:>7.3f} TF  "
                  f"{result['max_err']:>7.4f}")
    except Exception as e:
        print(f"seq={seq:>3d} d={d:>3d} ff={ff:>4d}  ERROR: {str(e)[:60]}")

ttnn.close_device(device)
print("\nDone!")
