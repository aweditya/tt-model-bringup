"""
Experiment 22: Trace capture for full transformer on Blackhole.

Now that on-device broadcast eliminates CPU round-trips, trace capture
should work for the full transformer Jaxpr. The approach:

1. Warmup run: execute Jaxpr normally, collecting literal tensors
2. Pre-load inputs and literals onto device
3. Trace capture: re-execute Jaxpr with cached literals (no host transfers)
4. Execute trace repeatedly for benchmark

Hypothesis: Trace capture eliminates Python dispatch overhead, giving
another ~2-5x speedup over interpreted execution (currently 348 fwd/sec).
"""

import sys, os
# On remote host, tt_jax lives at ~/tt_jax/ — parent dir must be on path
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import jax
import jax.numpy as jnp
from jax import make_jaxpr
import ttnn
import time

from tt_jax.interpret import Interpreter
from tt_jax import tensors

# ── Model ──────────────────────────────────────────────────────
def transformer_block(x, w_q, w_k, w_v, w_o, w1, w2, g1, b1, g2, b2):
    q = jnp.dot(x, w_q); k = jnp.dot(x, w_k); v = jnp.dot(x, w_v)
    scores = jnp.dot(q, k.T) / jnp.sqrt(jnp.array(64.0))
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

# ── Setup ──────────────────────────────────────────────────────
rng = np.random.RandomState(42)
s = 0.1
args = [
    rng.randn(32, 64).astype(np.float32) * s,   # x
    rng.randn(64, 64).astype(np.float32) * s,   # w_q
    rng.randn(64, 64).astype(np.float32) * s,   # w_k
    rng.randn(64, 64).astype(np.float32) * s,   # w_v
    rng.randn(64, 64).astype(np.float32) * s,   # w_o
    rng.randn(64, 256).astype(np.float32) * s,  # w1
    rng.randn(256, 64).astype(np.float32) * s,  # w2
    np.ones(64, dtype=np.float32),                # g1
    np.zeros(64, dtype=np.float32),               # b1
    np.ones(64, dtype=np.float32),                # g2
    np.zeros(64, dtype=np.float32),               # b2
]
jax_args = [jnp.array(a) for a in args]
jaxpr = make_jaxpr(transformer_block)(*jax_args)

print(f"Jaxpr: {len(jaxpr.jaxpr.eqns)} equations")
print(f"Unique ops: {sorted(set(e.primitive.name for e in jaxpr.jaxpr.eqns))}")

# ── Device ─────────────────────────────────────────────────────
device = ttnn.open_device(device_id=0)

# ── Step 1: Interpreted run (baseline) ─────────────────────────
print("\n=== Interpreted execution ===")
interp = Interpreter(device)
result = interp.run(jaxpr, args)
ref = np.array(transformer_block(*jax_args))
print(f"Max error: {np.abs(result - ref).max():.6f}")

for _ in range(3):
    interp.run(jaxpr, args)
N = 20
t0 = time.perf_counter()
for _ in range(N):
    interp.run(jaxpr, args)
t_interp = (time.perf_counter() - t0) / N
print(f"Interpreted: {t_interp*1000:.2f} ms ({1/t_interp:.0f} fwd/sec)")

# ── Step 2: Pre-materialize for trace capture ──────────────────
print("\n=== Preparing trace capture ===")

# Scan Jaxpr for all literal values that need to be on-device
literal_cache = {}
for eqn in jaxpr.jaxpr.eqns:
    for v in eqn.invars:
        if tensors.is_literal(v):
            fval = float(v.val)
            if fval not in literal_cache:
                literal_cache[fval] = tensors.to_device(v.val, device)
print(f"Pre-materialized {len(literal_cache)} literal(s): {sorted(literal_cache.keys())}")

# Pre-load inputs
input_tensors = [tensors.to_device(a, device) for a in args]
print(f"Pre-loaded {len(input_tensors)} input tensors")

# ── Step 3: Trace capture ─────────────────────────────────────
print("\n=== Trace capture ===")

# Create fresh interpreter with literal cache (no host transfers needed)
trace_interp = Interpreter(device, literal_cache=literal_cache)
for var, t in zip(jaxpr.jaxpr.invars, input_tensors):
    trace_interp.env[var] = t
# Bind constants
for var, const in zip(jaxpr.jaxpr.constvars, jaxpr.consts):
    trace_interp.env[var] = tensors.to_device(const, device)

try:
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    for eqn in jaxpr.jaxpr.eqns:
        trace_interp._exec(eqn)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    print(f"Trace capture SUCCEEDED! trace_id={tid}")
except Exception as e:
    print(f"Trace capture FAILED: {e}")
    ttnn.close_device(device)
    raise SystemExit(1)

# ── Step 4: Verify correctness ─────────────────────────────────
print("\n=== Verifying traced execution ===")
ttnn.execute_trace(device, tid, cq_id=0, blocking=True)

out_var = jaxpr.jaxpr.outvars[0]
out_shape = out_var.aval.shape
traced_result = tensors.from_device(trace_interp.env[out_var], out_shape)
traced_err = np.abs(traced_result - ref).max()
print(f"Traced max error: {traced_err:.6f}")

# ── Step 5: Benchmark traced execution ─────────────────────────
print("\n=== Benchmark ===")
for _ in range(10):
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)

N_trace = 500
t0 = time.perf_counter()
for _ in range(N_trace):
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
t_trace = (time.perf_counter() - t0) / N_trace

print(f"Traced:      {t_trace*1000:.3f} ms ({1/t_trace:.0f} fwd/sec)")
print(f"Interpreted: {t_interp*1000:.3f} ms ({1/t_interp:.0f} fwd/sec)")
print(f"Speedup:     {t_interp/t_trace:.1f}x")

ttnn.release_trace(device, tid)
ttnn.close_device(device)

print("\nDone!")
