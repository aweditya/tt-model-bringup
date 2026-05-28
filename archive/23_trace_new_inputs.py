"""
Experiment 23: Trace capture with changeable inputs.

The key question: can we feed NEW data to a traced computation?

TT-NN trace capture records device operations on specific buffer addresses.
To change inputs, we use ttnn.copy to write new data into the SAME device
buffers that the trace was captured with. The trace then replays on the new data.

Critical insight: trace capture requires a WARMUP run first to allocate all
intermediate buffers. The trace then reuses those exact buffers — no new
allocations during capture.

This is how a real backend would work:
  1. Compile: warmup run + trace capture (one-time cost)
  2. Execute: ttnn.copy new inputs → execute_trace (per-inference)
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
    rng.randn(32, 64).astype(np.float32) * s,   # x (this changes each inference)
    rng.randn(64, 64).astype(np.float32) * s,   # w_q (weights are fixed)
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

device = ttnn.open_device(device_id=0)

# ── Pre-materialize literals ──────────────────────────────────
literal_cache = {}
for eqn in jaxpr.jaxpr.eqns:
    for v in eqn.invars:
        if tensors.is_literal(v):
            fval = float(v.val)
            if fval not in literal_cache:
                literal_cache[fval] = tensors.to_device(v.val, device)

# ── Pre-load inputs ───────────────────────────────────────────
input_tensors = [tensors.to_device(a, device) for a in args]
x_tt = input_tensors[0]  # This is the buffer we'll overwrite

# ── Step 1: Warmup run (allocates all intermediate buffers) ───
print("\n=== Warmup ===")
interp = Interpreter(device, literal_cache=literal_cache)
for var, t in zip(jaxpr.jaxpr.invars, input_tensors):
    interp.env[var] = t
for eqn in jaxpr.jaxpr.eqns:
    interp._exec(eqn)
out_var = jaxpr.jaxpr.outvars[0]
print("Warmup complete — all buffers allocated")

# ── Step 2: Trace capture (reuses existing buffers) ───────────
print("\n=== Trace capture ===")
tid = ttnn.begin_trace_capture(device, cq_id=0)
for eqn in jaxpr.jaxpr.eqns:
    interp._exec(eqn)
ttnn.end_trace_capture(device, tid, cq_id=0)
print(f"Trace captured! tid={tid}")

# ── Step 3: Verify with same input ───────────────────────────
print("\n=== Same input ===")
ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
result1 = tensors.from_device(interp.env[out_var], out_var.aval.shape)
ref1 = np.array(transformer_block(*jax_args))
print(f"Max error: {np.abs(result1 - ref1).max():.6f}")

# ── Step 4: New input via ttnn.copy ───────────────────────────
print("\n=== New input via ttnn.copy ===")
x_np2 = rng.randn(32, 64).astype(np.float32) * 0.3
x_tt2 = tensors.to_device(x_np2, device)
ttnn.copy(x_tt2, x_tt)  # Overwrite the input buffer
ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
result2 = tensors.from_device(interp.env[out_var], out_var.aval.shape)
ref2 = np.array(transformer_block(jnp.array(x_np2), *jax_args[1:]))
err2 = np.abs(result2 - ref2).max()
print(f"Max error (new input): {err2:.6f}")
if err2 < 0.1:
    print("SUCCESS: Trace works with new inputs!")

# ── Step 5: Benchmark ────────────────────────────────────────
print("\n=== Benchmark ===")
x_inputs_tt = [tensors.to_device(rng.randn(32, 64).astype(np.float32) * s, device)
               for _ in range(50)]

# Warmup
for xi in x_inputs_tt[:5]:
    ttnn.copy(xi, x_tt)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)

# Trace only (same input, no copy)
N = 500
t0 = time.perf_counter()
for _ in range(N):
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
t_trace = (time.perf_counter() - t0) / N

# Copy + trace (different inputs)
N2 = 200
t0 = time.perf_counter()
for i in range(N2):
    ttnn.copy(x_inputs_tt[i % len(x_inputs_tt)], x_tt)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
t_copy_trace = (time.perf_counter() - t0) / N2

print(f"Trace only:       {t_trace*1000:.3f} ms ({1/t_trace:.0f} fwd/sec)")
print(f"Copy + trace:     {t_copy_trace*1000:.3f} ms ({1/t_copy_trace:.0f} fwd/sec)")
print(f"Copy overhead:    {(t_copy_trace - t_trace)*1000:.3f} ms")

ttnn.release_trace(device, tid)
ttnn.close_device(device)
print("\nDone!")
