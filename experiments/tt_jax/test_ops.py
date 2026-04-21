"""
Unit tests for the tt_jax op registry.

Each test traces a minimal JAX function to isolate a single Jaxpr primitive,
then verifies the TT-NN implementation matches the JAX CPU reference.

Run on remote host:
    python3 -m pytest test_ops.py -v
    OR
    python3 test_ops.py  (standalone runner at bottom)
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import make_jaxpr
import ttnn
import sys
import os

# Add parent dir to path so we can import tt_jax
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tt_jax.interpret import Interpreter


# ============================================================
# Test infrastructure
# ============================================================

_device = None

def get_device():
    global _device
    if _device is None:
        _device = ttnn.open_device(device_id=0)
    return _device

def assert_close(result, reference, name, max_atol=0.2, mean_atol=0.05):
    """Assert result matches reference within bf16 tolerance."""
    err = np.abs(result - reference)
    max_err = err.max()
    mean_err = err.mean()
    passed = max_err < max_atol and mean_err < mean_atol
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}: max_err={max_err:.6f} mean_err={mean_err:.6f}")
    if not passed:
        print(f"         Expected max<{max_atol}, mean<{mean_atol}")
    return passed

def run_jaxpr_test(name, fn, args, max_atol=0.2, mean_atol=0.05):
    """Trace fn to Jaxpr, interpret on Blackhole, compare to JAX CPU."""
    jax_args = [jnp.array(a) for a in args]
    ref = np.array(fn(*jax_args))
    jaxpr = make_jaxpr(fn)(*jax_args)

    interp = Interpreter(get_device())
    result = interp.run(jaxpr, args)

    return assert_close(result, ref, name, max_atol, mean_atol)


# ============================================================
# Tests: Elementwise ops
# ============================================================

def test_add():
    x = np.random.randn(32, 64).astype(np.float32)
    y = np.random.randn(32, 64).astype(np.float32)
    return run_jaxpr_test("add(x, y)", lambda a, b: a + b, [x, y])

def test_add_scalar():
    x = np.random.randn(32, 64).astype(np.float32)
    return run_jaxpr_test("add(x, 3.0)", lambda a: a + 3.0, [x])

def test_sub():
    x = np.random.randn(32, 64).astype(np.float32)
    y = np.random.randn(32, 64).astype(np.float32)
    return run_jaxpr_test("sub(x, y)", lambda a, b: a - b, [x, y])

def test_mul():
    x = np.random.randn(32, 64).astype(np.float32)
    y = np.random.randn(32, 64).astype(np.float32)
    return run_jaxpr_test("mul(x, y)", lambda a, b: a * b, [x, y])

def test_mul_scalar():
    x = np.random.randn(32, 64).astype(np.float32)
    return run_jaxpr_test("mul(x, 2.5)", lambda a: a * 2.5, [x])

def test_neg():
    x = np.random.randn(32, 64).astype(np.float32)
    return run_jaxpr_test("neg(x)", lambda a: -a, [x])

def test_exp():
    x = np.random.randn(32, 64).astype(np.float32) * 0.5  # small range
    return run_jaxpr_test("exp(x)", lambda a: jnp.exp(a), [x], max_atol=0.3)

def test_log():
    x = np.abs(np.random.randn(32, 64).astype(np.float32)) + 0.1
    return run_jaxpr_test("log(x)", lambda a: jnp.log(a), [x])

def test_sqrt():
    x = np.abs(np.random.randn(32, 64).astype(np.float32)) + 0.1
    return run_jaxpr_test("sqrt(x)", lambda a: jnp.sqrt(a), [x])

def test_relu():
    x = np.random.randn(32, 64).astype(np.float32)
    return run_jaxpr_test("relu(x)", lambda a: jax.nn.relu(a), [x])

def test_square():
    x = np.random.randn(32, 64).astype(np.float32)
    return run_jaxpr_test("x^2", lambda a: a ** 2, [x], max_atol=0.5)


# ============================================================
# Tests: Matmul
# ============================================================

def test_matmul():
    x = np.random.randn(32, 64).astype(np.float32) * 0.1
    w = np.random.randn(64, 32).astype(np.float32) * 0.1
    return run_jaxpr_test("matmul(x, w)", lambda a, b: jnp.dot(a, b), [x, w],
                          max_atol=0.5, mean_atol=0.1)

def test_matmul_add():
    x = np.random.randn(32, 64).astype(np.float32) * 0.1
    w = np.random.randn(64, 32).astype(np.float32) * 0.1
    b = np.random.randn(32).astype(np.float32) * 0.01
    return run_jaxpr_test("matmul+bias",
                          lambda a, b, c: jnp.dot(a, b) + c, [x, w, b],
                          max_atol=0.5, mean_atol=0.1)


# ============================================================
# Tests: Reductions
# ============================================================

def test_reduce_sum():
    x = np.random.randn(32, 64).astype(np.float32)
    return run_jaxpr_test("sum(x, axis=-1)",
                          lambda a: jnp.sum(a, axis=-1), [x],
                          max_atol=1.0, mean_atol=0.5)

def test_reduce_max():
    x = np.random.randn(32, 64).astype(np.float32)
    return run_jaxpr_test("max(x, axis=-1)",
                          lambda a: jnp.max(a, axis=-1), [x])


# ============================================================
# Tests: Composite functions
# ============================================================

def test_softmax():
    x = np.random.randn(32, 64).astype(np.float32)
    return run_jaxpr_test("softmax(x)",
                          lambda a: jax.nn.softmax(a, axis=-1), [x],
                          max_atol=0.01, mean_atol=0.001)

def test_layer_norm():
    x = np.random.randn(32, 64).astype(np.float32)
    g = np.ones(64, dtype=np.float32)
    b = np.zeros(64, dtype=np.float32)
    def ln(x, g, b):
        m = jnp.mean(x, axis=-1, keepdims=True)
        v = jnp.mean((x - m) ** 2, axis=-1, keepdims=True)
        return g * (x - m) / jnp.sqrt(v + 1e-5) + b
    return run_jaxpr_test("layer_norm(x)", ln, [x, g, b],
                          max_atol=0.1, mean_atol=0.01)

def test_mlp():
    x = np.random.randn(32, 64).astype(np.float32) * 0.1
    w1 = np.random.randn(64, 128).astype(np.float32) * 0.1
    b1 = np.zeros(128, dtype=np.float32)
    w2 = np.random.randn(128, 32).astype(np.float32) * 0.1
    b2 = np.zeros(32, dtype=np.float32)
    def mlp(x, w1, b1, w2, b2):
        h = jax.nn.relu(jnp.dot(x, w1) + b1)
        return jnp.dot(h, w2) + b2
    return run_jaxpr_test("MLP(x)", mlp, [x, w1, b1, w2, b2],
                          max_atol=1.0, mean_atol=0.1)

def test_linear_layernorm():
    """Linear + ReLU + LayerNorm — a transformer sub-block."""
    x = np.random.randn(32, 64).astype(np.float32) * 0.1
    w = np.random.randn(64, 64).astype(np.float32) * 0.1
    b = np.zeros(64, dtype=np.float32)
    g = np.ones(64, dtype=np.float32)
    bt = np.zeros(64, dtype=np.float32)
    def block(x, w, b, g, bt):
        h = jax.nn.relu(jnp.dot(x, w) + b)
        m = jnp.mean(h, axis=-1, keepdims=True)
        v = jnp.mean((h - m) ** 2, axis=-1, keepdims=True)
        return g * (h - m) / jnp.sqrt(v + 1e-5) + bt
    return run_jaxpr_test("Linear+ReLU+LayerNorm", block,
                          [x, w, b, g, bt],
                          max_atol=0.2, mean_atol=0.02)


# ============================================================
# Standalone runner
# ============================================================

ALL_TESTS = [
    # Elementwise
    test_add, test_add_scalar, test_sub, test_mul, test_mul_scalar,
    test_neg, test_exp, test_log, test_sqrt, test_relu, test_square,
    # Matmul
    test_matmul, test_matmul_add,
    # Reductions
    test_reduce_sum, test_reduce_max,
    # Composite
    test_softmax, test_layer_norm, test_mlp, test_linear_layernorm,
]

if __name__ == '__main__':
    np.random.seed(42)
    print(f"tt_jax unit tests")
    print(f"{'=' * 60}")

    passed = 0
    failed = 0
    errors = []

    for test_fn in ALL_TESTS:
        try:
            ok = test_fn()
            if ok:
                passed += 1
            else:
                failed += 1
                errors.append(test_fn.__name__)
        except Exception as e:
            failed += 1
            errors.append(f"{test_fn.__name__}: {type(e).__name__}: {str(e)[:80]}")
            print(f"  [ERROR] {test_fn.__name__}: {type(e).__name__}: {str(e)[:80]}")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(ALL_TESTS)}")
    if errors:
        print(f"\nFailed tests:")
        for e in errors:
            print(f"  - {e}")

    ttnn.close_device(get_device())
    sys.exit(0 if failed == 0 else 1)
