"""
Experiment 20: Full transformer encoder block through tt_jax interpreter.

Tests:
1. Trace a JAX transformer to Jaxpr and check which primitives it uses
2. Run through the tt_jax interpreter on Blackhole
3. Compare against JAX CPU reference
4. Benchmark traced vs non-traced execution

Run: python3 20_transformer_via_jaxpr.py
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import make_jaxpr
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tt_jax.interpret import Interpreter
from tt_jax.trace import TracedExecutor
import ttnn


# ============================================================
# Step 1: Define transformer in pure JAX and trace to Jaxpr
# ============================================================

def transformer_block(x, w_q, w_k, w_v, w_o, w1, w2, g1, b1, g2, b2):
    """Single-head transformer encoder block in pure JAX."""
    # Self-attention
    q = jnp.dot(x, w_q)
    k = jnp.dot(x, w_k)
    v = jnp.dot(x, w_v)

    # Scaled dot-product attention
    scores = jnp.dot(q, k.T) / jnp.sqrt(jnp.array(64.0))
    attn = jax.nn.softmax(scores, axis=-1)
    context = jnp.dot(attn, v)

    # Output projection + residual
    h = x + jnp.dot(context, w_o)

    # Layer norm 1
    m1 = jnp.mean(h, axis=-1, keepdims=True)
    v1 = jnp.mean((h - m1) ** 2, axis=-1, keepdims=True)
    h = g1 * (h - m1) / jnp.sqrt(v1 + 1e-5) + b1

    # FFN: relu(xW1)W2
    ff = jax.nn.relu(jnp.dot(h, w1))
    ff = jnp.dot(ff, w2)
    h2 = h + ff

    # Layer norm 2
    m2 = jnp.mean(h2, axis=-1, keepdims=True)
    v2 = jnp.mean((h2 - m2) ** 2, axis=-1, keepdims=True)
    out = g2 * (h2 - m2) / jnp.sqrt(v2 + 1e-5) + b2

    return out


def make_args(seq_len=32, d_model=64, d_ff=256, seed=42):
    """Create random transformer weights."""
    rng = np.random.RandomState(seed)
    scale = 0.1
    return [
        rng.randn(seq_len, d_model).astype(np.float32) * scale,  # x
        rng.randn(d_model, d_model).astype(np.float32) * scale,  # w_q
        rng.randn(d_model, d_model).astype(np.float32) * scale,  # w_k
        rng.randn(d_model, d_model).astype(np.float32) * scale,  # w_v
        rng.randn(d_model, d_model).astype(np.float32) * scale,  # w_o
        rng.randn(d_model, d_ff).astype(np.float32) * scale,     # w1
        rng.randn(d_ff, d_model).astype(np.float32) * scale,     # w2
        np.ones(d_model, dtype=np.float32),                        # g1
        np.zeros(d_model, dtype=np.float32),                       # b1
        np.ones(d_model, dtype=np.float32),                        # g2
        np.zeros(d_model, dtype=np.float32),                       # b2
    ]


def main():
    print("=" * 70)
    print("Experiment 20: Transformer encoder via tt_jax Jaxpr interpreter")
    print("=" * 70)

    # ---- Step 1: Trace and analyze ----
    print("\n--- Step 1: Trace transformer to Jaxpr ---")
    args = make_args()
    jax_args = [jnp.array(a) for a in args]
    jaxpr = make_jaxpr(transformer_block)(*jax_args)

    # Collect primitives
    prims = set()
    def collect(jaxpr_inner):
        for eqn in jaxpr_inner.eqns:
            prims.add(eqn.primitive.name)
            for p in eqn.params.values():
                if hasattr(p, 'jaxpr'):
                    sub = p.jaxpr if hasattr(p.jaxpr, 'eqns') else p
                    collect(sub if hasattr(sub, 'eqns') else sub)
                elif hasattr(p, 'eqns'):
                    collect(p)

    collect(jaxpr.jaxpr)
    print(f"Total Jaxpr equations: {len(jaxpr.jaxpr.eqns)}")
    print(f"Unique primitives: {sorted(prims)}")

    supported = {
        'add', 'sub', 'mul', 'div', 'neg', 'exp', 'log', 'sqrt', 'rsqrt',
        'reciprocal', 'max', 'integer_pow', 'dot_general', 'reduce_max',
        'reduce_sum', 'broadcast_in_dim', 'reshape', 'transpose', 'squeeze',
        'convert_element_type', 'stop_gradient',
    }
    sub_jaxpr_ops = {'custom_jvp_call', 'pjit'}
    missing = prims - supported - sub_jaxpr_ops
    print(f"Supported: {len(prims & (supported | sub_jaxpr_ops))}/{len(prims)}")
    if missing:
        print(f"MISSING: {sorted(missing)}")
        print("Cannot proceed until these are implemented.")
        return
    print("All primitives supported!")

    # ---- Step 2: JAX CPU reference ----
    print("\n--- Step 2: JAX CPU reference ---")
    ref = np.array(transformer_block(*jax_args))
    print(f"Output shape: {ref.shape}")
    print(f"Output range: [{ref.min():.4f}, {ref.max():.4f}]")

    # ---- Step 3: Run on Blackhole via interpreter ----
    print("\n--- Step 3: Run on Blackhole via tt_jax interpreter ---")
    device = ttnn.open_device(device_id=0)

    try:
        interp = Interpreter(device)
        result = interp.run(jaxpr, args)

        err = np.abs(result - ref)
        max_err = err.max()
        mean_err = err.mean()
        print(f"Max error:  {max_err:.6f}")
        print(f"Mean error: {mean_err:.6f}")

        if max_err < 0.5 and mean_err < 0.05:
            print("PASS: Transformer output matches JAX CPU reference!")
        else:
            print("FAIL: Error too large")
            print(f"  Expected max<0.5, mean<0.05")

        # ---- Step 4: Benchmark interpreted execution ----
        print("\n--- Step 4: Benchmark interpreted execution ---")
        # Warmup
        for _ in range(3):
            interp.run(jaxpr, args)

        N = 20
        t0 = time.perf_counter()
        for _ in range(N):
            interp.run(jaxpr, args)
        t_interp = (time.perf_counter() - t0) / N
        print(f"Interpreted: {t_interp*1000:.2f} ms/forward")

        # ---- Step 5: Benchmark traced execution ----
        print("\n--- Step 5: Benchmark traced execution ---")
        executor = TracedExecutor(device)
        executor.compile(jaxpr, args)

        # Verify traced output
        traced_result = executor.run(args)
        traced_err = np.abs(traced_result - ref)
        print(f"Traced max error:  {traced_err.max():.6f}")
        print(f"Traced mean error: {traced_err.mean():.6f}")

        # Warmup
        for _ in range(5):
            executor.run(args)

        N_trace = 100
        t0 = time.perf_counter()
        for _ in range(N_trace):
            executor.run(args)
        t_traced = (time.perf_counter() - t0) / N_trace
        print(f"Traced:      {t_traced*1000:.2f} ms/forward")
        print(f"Speedup:     {t_interp/t_traced:.2f}x")

        executor.release()

        # ---- Step 6: Profile op breakdown ----
        print("\n--- Step 6: Op frequency breakdown ---")
        from collections import Counter
        op_counts = Counter()
        def count_ops(jaxpr_inner):
            for eqn in jaxpr_inner.eqns:
                op_counts[eqn.primitive.name] += 1
        count_ops(jaxpr.jaxpr)
        print(f"{'Op':<25} {'Count':>5}")
        print("-" * 32)
        for op, count in op_counts.most_common():
            print(f"{op:<25} {count:>5}")

    finally:
        ttnn.close_device(device)

    print("\n" + "=" * 70)
    print("Experiment 20 complete!")


if __name__ == '__main__':
    main()
