"""
Experiment 25: Multi-layer transformer on Blackhole.

Can we stack multiple transformer layers and maintain correctness
and good throughput? This tests the system at a more realistic scale.

We'll use a loop that applies the same weights N times (weight-tied
layers) to keep parameter count manageable.
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


def single_layer(x, w_q, w_k, w_v, w_o, w1, w2, g1, b1, g2, b2):
    """One transformer encoder layer."""
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
    return g2 * (h2 - m2) / jnp.sqrt(v2 + 1e-5) + b2


def multi_layer(x, w_q, w_k, w_v, w_o, w1, w2, g1, b1, g2, b2, n_layers):
    """N-layer transformer with tied weights."""
    for _ in range(n_layers):
        x = single_layer(x, w_q, w_k, w_v, w_o, w1, w2, g1, b1, g2, b2)
    return x


device = ttnn.open_device(device_id=0)

seq_len = 64
d_model = 128
d_ff = 512

rng = np.random.RandomState(42)
s = 0.01
weight_args = [
    rng.randn(d_model, d_model).astype(np.float32) * s,  # w_q
    rng.randn(d_model, d_model).astype(np.float32) * s,  # w_k
    rng.randn(d_model, d_model).astype(np.float32) * s,  # w_v
    rng.randn(d_model, d_model).astype(np.float32) * s,  # w_o
    rng.randn(d_model, d_ff).astype(np.float32) * s,     # w1
    rng.randn(d_ff, d_model).astype(np.float32) * s,     # w2
    np.ones(d_model, dtype=np.float32),                    # g1
    np.zeros(d_model, dtype=np.float32),                   # b1
    np.ones(d_model, dtype=np.float32),                    # g2
    np.zeros(d_model, dtype=np.float32),                   # b2
]
x_np = rng.randn(seq_len, d_model).astype(np.float32) * s

print(f"Config: seq={seq_len}, d_model={d_model}, d_ff={d_ff}")
print(f"{'Layers':>6s}  {'Eqns':>5s}  {'Trace ms':>9s}  {'fwd/sec':>8s}  {'Error':>8s}")
print("-" * 45)

for n_layers in [1, 2, 4, 6, 8, 12]:
    def model(x, *weights):
        return multi_layer(x, *weights, n_layers=n_layers)

    jax_x = jnp.array(x_np)
    jax_weights = [jnp.array(w) for w in weight_args]

    try:
        jaxpr = make_jaxpr(model)(jax_x, *jax_weights)
        n_eqns = len(jaxpr.jaxpr.eqns)

        # Pre-materialize literals
        literal_cache = {}
        for eqn in jaxpr.jaxpr.eqns:
            for v in eqn.invars:
                if tensors.is_literal(v):
                    fval = float(v.val)
                    if fval not in literal_cache:
                        literal_cache[fval] = tensors.to_device(v.val, device)

        all_args = [x_np] + weight_args
        input_tensors = [tensors.to_device(a, device) for a in all_args]

        # Warmup
        interp = Interpreter(device, literal_cache=literal_cache)
        for var, t in zip(jaxpr.jaxpr.invars, input_tensors):
            interp.env[var] = t
        for eqn in jaxpr.jaxpr.eqns:
            interp._exec(eqn)

        # Trace
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        for eqn in jaxpr.jaxpr.eqns:
            interp._exec(eqn)
        ttnn.end_trace_capture(device, tid, cq_id=0)

        # Verify
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        out_var = jaxpr.jaxpr.outvars[0]
        result = tensors.from_device(interp.env[out_var], out_var.aval.shape)
        ref = np.array(model(jax_x, *jax_weights))
        max_err = np.abs(result - ref).max()

        # Benchmark
        for _ in range(10):
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        N = max(50, 200 // n_layers)
        t0 = time.perf_counter()
        for _ in range(N):
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        t_avg = (time.perf_counter() - t0) / N

        print(f"{n_layers:>6d}  {n_eqns:>5d}  {t_avg*1000:>8.3f}   {1/t_avg:>7.0f}/s  {max_err:>7.4f}")

        ttnn.release_trace(device, tid)

    except Exception as e:
        print(f"{n_layers:>6d}  ERROR: {str(e)[:60]}")

ttnn.close_device(device)
print("\nDone!")
