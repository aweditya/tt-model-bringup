"""Test the StableHLO Python interpreter engine.

Tests that the engine correctly parses MLIR bytecode and executes
StableHLO ops on numpy arrays. These tests validate the Python engine
independently of the C++ PJRT plugin.

NOTE: Example args use np.ones (not jnp.ones) to avoid device placement
when the TT plugin is loaded. Lowering is forced to CPU in get_bytecode.
"""

import sys
import os

import pytest
import numpy as np

# Add plugin dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jax_plugins.tt.engine import (
    bytecode_to_text,
    parse_stablehlo,
    execute_stablehlo,
    parse_tensor_type,
    execute_op,
)


# ============================================================
# Helpers to get real StableHLO bytecode from JAX
# ============================================================

def get_bytecode(fn, *example_args):
    """JIT-lower a function and return its StableHLO bytecode.

    Uses CPU for lowering so engine tests work regardless of which
    backends are registered (TT plugin may or may not be loaded).
    """
    import jax
    import jax._src.interpreters.mlir as jax_mlir

    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        lowered = jax.jit(fn).lower(*example_args)
        module = lowered.compiler_ir(dialect="stablehlo")
        return jax_mlir.module_to_bytecode(module)


# ============================================================
# Parser tests
# ============================================================

class TestParser:
    def test_vhlo_portable_artifact(self):
        """Test parsing VHLO portable artifacts (what JAX sends to PJRT)."""
        import jax
        from jaxlib.mlir._mlir_libs._stablehlo import (
            serialize_portable_artifact, get_current_version
        )

        cpu = jax.devices("cpu")[0]
        with jax.default_device(cpu):
            f = jax.jit(lambda x: x + 1.0)
            lowered = f.lower(np.ones(4, dtype=np.float32))
            module = lowered.compiler_ir(dialect="stablehlo")

        # Serialize as portable artifact (same format JAX sends to PJRT)
        version = get_current_version()
        portable = serialize_portable_artifact(module, version)
        assert b'StableHLO' in portable[:30]

        # Our engine should handle this format
        text = bytecode_to_text(portable)
        assert 'stablehlo.add' in text
        assert 'stablehlo.constant' in text

        # Execute should work too
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        [result] = execute_stablehlo(portable, [x])
        np.testing.assert_allclose(result, x + 1.0)

    def test_parse_tensor_type_2d(self):
        shape, dtype = parse_tensor_type("tensor<2x3xf32>")
        assert shape == (2, 3)
        assert dtype == np.float32

    def test_parse_tensor_type_1d(self):
        shape, dtype = parse_tensor_type("tensor<4xf32>")
        assert shape == (4,)
        assert dtype == np.float32

    def test_parse_tensor_type_scalar(self):
        shape, dtype = parse_tensor_type("tensor<f32>")
        assert shape == ()
        assert dtype == np.float32

    def test_parse_tensor_type_int32(self):
        shape, dtype = parse_tensor_type("tensor<3xi32>")
        assert shape == (3,)
        assert dtype == np.int32

    def test_bytecode_to_text(self):
        import jax.numpy as jnp
        bc = get_bytecode(lambda x: x + 1.0, np.ones(4, dtype=np.float32))
        text = bytecode_to_text(bc)
        assert "stablehlo.add" in text
        assert "stablehlo.constant" in text

    def test_parse_add_program(self):
        import jax.numpy as jnp
        bc = get_bytecode(lambda x: x + 1.0, np.ones(4, dtype=np.float32))
        text = bytecode_to_text(bc)
        args, ops, returns, _private = parse_stablehlo(text)

        assert len(args) == 1
        assert args[0][0] == "arg0"
        assert len(returns) >= 1

        op_types = [op['op'] for op in ops]
        assert 'constant' in op_types
        assert 'add' in op_types

    def test_parse_matmul_program(self):
        import jax.numpy as jnp
        bc = get_bytecode(lambda x, w: x @ w, np.ones((2, 3), dtype=np.float32), np.ones((3, 4), dtype=np.float32))
        text = bytecode_to_text(bc)
        args, ops, returns, _private = parse_stablehlo(text)

        assert len(args) == 2
        op_types = [op['op'] for op in ops]
        assert 'dot_general' in op_types


# ============================================================
# Execution tests
# ============================================================

class TestExecution:
    def test_add_scalar(self):
        """x + 1.0"""
        import jax.numpy as jnp
        bc = get_bytecode(lambda x: x + 1.0, np.ones(4, dtype=np.float32))
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        [result] = execute_stablehlo(bc, [x])
        np.testing.assert_allclose(result, x + 1.0)

    def test_multiply_add(self):
        """x * 2 + 3"""
        import jax.numpy as jnp
        bc = get_bytecode(lambda x: x * 2.0 + 3.0, np.ones(4, dtype=np.float32))
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        [result] = execute_stablehlo(bc, [x])
        np.testing.assert_allclose(result, x * 2.0 + 3.0)

    def test_subtract(self):
        """x - y"""
        import jax.numpy as jnp
        bc = get_bytecode(lambda x, y: x - y, np.ones(4, dtype=np.float32), np.ones(4, dtype=np.float32))
        x = np.array([5.0, 4.0, 3.0, 2.0], dtype=np.float32)
        y = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        [result] = execute_stablehlo(bc, [x, y])
        np.testing.assert_allclose(result, x - y)

    def test_negate(self):
        """-x"""
        import jax.numpy as jnp
        bc = get_bytecode(lambda x: -x, np.ones(4, dtype=np.float32))
        x = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
        [result] = execute_stablehlo(bc, [x])
        np.testing.assert_allclose(result, -x)

    def test_exp(self):
        """exp(x)"""
        import jax.numpy as jnp
        bc = get_bytecode(lambda x: jnp.exp(x), np.ones(4, dtype=np.float32))
        x = np.array([0.0, 1.0, 2.0, -1.0], dtype=np.float32)
        [result] = execute_stablehlo(bc, [x])
        np.testing.assert_allclose(result, np.exp(x), rtol=1e-6)

    def test_tanh(self):
        """tanh(x)"""
        import jax.numpy as jnp
        bc = get_bytecode(lambda x: jnp.tanh(x), np.ones(4, dtype=np.float32))
        x = np.array([0.0, 1.0, -1.0, 3.0], dtype=np.float32)
        [result] = execute_stablehlo(bc, [x])
        np.testing.assert_allclose(result, np.tanh(x), rtol=1e-6)

    def test_matmul(self):
        """x @ w"""
        import jax.numpy as jnp
        bc = get_bytecode(lambda x, w: x @ w, np.ones((2, 3), dtype=np.float32), np.ones((3, 4), dtype=np.float32))
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        w = np.eye(3, 4, dtype=np.float32) * 2
        [result] = execute_stablehlo(bc, [x, w])
        np.testing.assert_allclose(result, x @ w)

    def test_matmul_add(self):
        """x @ w + b (linear layer)"""
        import jax.numpy as jnp
        bc = get_bytecode(
            lambda x, w, b: x @ w + b,
            np.ones((2, 3), dtype=np.float32), np.ones((3, 4), dtype=np.float32), np.ones(4, dtype=np.float32)
        )
        x = np.random.randn(2, 3).astype(np.float32)
        w = np.random.randn(3, 4).astype(np.float32)
        b = np.random.randn(4).astype(np.float32)
        [result] = execute_stablehlo(bc, [x, w, b])
        np.testing.assert_allclose(result, x @ w + b, rtol=1e-5)

    def test_larger_array(self):
        """Larger array round-trip through computation."""
        import jax.numpy as jnp
        bc = get_bytecode(lambda x: x * 3.14 + 2.71, np.ones((32, 64), dtype=np.float32))
        x = np.random.randn(32, 64).astype(np.float32)
        [result] = execute_stablehlo(bc, [x])
        np.testing.assert_allclose(result, x * 3.14 + 2.71, rtol=1e-5)

    def test_element_multiply(self):
        """x * y element-wise"""
        import jax.numpy as jnp
        bc = get_bytecode(lambda x, y: x * y, np.ones(4, dtype=np.float32), np.ones(4, dtype=np.float32))
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        y = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        [result] = execute_stablehlo(bc, [x, y])
        np.testing.assert_allclose(result, x * y)

    def test_divide(self):
        """x / y"""
        import jax.numpy as jnp
        bc = get_bytecode(lambda x, y: x / y, np.ones(4, dtype=np.float32), np.ones(4, dtype=np.float32))
        x = np.array([6.0, 8.0, 9.0, 12.0], dtype=np.float32)
        y = np.array([2.0, 4.0, 3.0, 6.0], dtype=np.float32)
        [result] = execute_stablehlo(bc, [x, y])
        np.testing.assert_allclose(result, x / y)

    def test_maximum(self):
        """max(x, 0) (relu)"""
        import jax.numpy as jnp
        bc = get_bytecode(lambda x: jnp.maximum(x, 0.0), np.ones(4, dtype=np.float32))
        x = np.array([-2.0, -1.0, 0.0, 1.0], dtype=np.float32)
        [result] = execute_stablehlo(bc, [x])
        np.testing.assert_allclose(result, np.maximum(x, 0.0))


# ============================================================
# Tests: Reduce ops
# ============================================================

class TestReduce:
    def test_reduce_sum(self):
        """sum(x, axis=-1)"""
        import jax.numpy as jnp
        bc = get_bytecode(lambda x: jnp.sum(x, axis=-1), np.ones((4, 8), dtype=np.float32))
        x = np.random.randn(4, 8).astype(np.float32)
        [result] = execute_stablehlo(bc, [x])
        np.testing.assert_allclose(result, np.sum(x, axis=-1), rtol=1e-5)

    def test_reduce_max(self):
        """max(x, axis=-1)"""
        import jax.numpy as jnp
        bc = get_bytecode(lambda x: jnp.max(x, axis=-1), np.ones((4, 8), dtype=np.float32))
        x = np.random.randn(4, 8).astype(np.float32)
        [result] = execute_stablehlo(bc, [x])
        np.testing.assert_allclose(result, np.max(x, axis=-1))

    def test_reduce_sum_keepdims(self):
        """sum with keepdims via mean pattern (sum / N)"""
        import jax.numpy as jnp
        bc = get_bytecode(
            lambda x: jnp.mean(x, axis=-1, keepdims=True),
            np.ones((2, 64), dtype=np.float32),
        )
        x = np.random.randn(2, 64).astype(np.float32)
        [result] = execute_stablehlo(bc, [x])
        np.testing.assert_allclose(
            result, np.mean(x, axis=-1, keepdims=True), rtol=1e-5
        )


# ============================================================
# Tests: Composite functions (require reduce)
# ============================================================

class TestComposite:
    def test_softmax(self):
        """jax.nn.softmax end-to-end through engine"""
        import jax
        import jax.numpy as jnp
        bc = get_bytecode(
            lambda x: jax.nn.softmax(x, axis=-1),
            np.ones((2, 64), dtype=np.float32),
        )
        x = np.random.randn(2, 64).astype(np.float32)
        [result] = execute_stablehlo(bc, [x])
        from scipy.special import softmax as scipy_softmax
        # Use numpy softmax reference instead of scipy to avoid dependency
        ref = np.exp(x - np.max(x, axis=-1, keepdims=True))
        ref = ref / np.sum(ref, axis=-1, keepdims=True)
        np.testing.assert_allclose(result, ref, rtol=1e-5)

    def test_layer_norm(self):
        """Manual layer norm through engine"""
        import jax.numpy as jnp
        def layer_norm(x, g, b):
            mean = jnp.mean(x, axis=-1, keepdims=True)
            var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
            return g * (x - mean) / jnp.sqrt(var + 1e-5) + b
        bc = get_bytecode(
            layer_norm,
            np.ones((2, 64), dtype=np.float32),
            np.ones(64, dtype=np.float32),
            np.zeros(64, dtype=np.float32),
        )
        x = np.random.randn(2, 64).astype(np.float32)
        g = np.ones(64, dtype=np.float32)
        b = np.zeros(64, dtype=np.float32)
        [result] = execute_stablehlo(bc, [x, g, b])
        # Numpy reference
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
        ref = g * (x - mean) / np.sqrt(var + 1e-5) + b
        np.testing.assert_allclose(result, ref, rtol=1e-5)

    def test_rms_norm(self):
        """RMS norm (Llama/Qwen style) through engine"""
        import jax.numpy as jnp
        def rms_norm(x, g):
            ms = jnp.mean(x ** 2, axis=-1, keepdims=True)
            return g * x / jnp.sqrt(ms + 1e-6)
        bc = get_bytecode(
            rms_norm,
            np.ones((2, 64), dtype=np.float32),
            np.ones(64, dtype=np.float32),
        )
        x = np.random.randn(2, 64).astype(np.float32)
        g = np.ones(64, dtype=np.float32)
        [result] = execute_stablehlo(bc, [x, g])
        ms = np.mean(x ** 2, axis=-1, keepdims=True)
        ref = g * x / np.sqrt(ms + 1e-6)
        np.testing.assert_allclose(result, ref, rtol=1e-5)

    def test_attention(self):
        """Single-head self-attention through engine"""
        import jax
        import jax.numpy as jnp
        def attention(x, wq, wk, wv, wo):
            q = x @ wq
            k = x @ wk
            v = x @ wv
            d = jnp.float32(q.shape[-1])
            scores = jax.nn.softmax(q @ k.T / jnp.sqrt(d), axis=-1)
            return (scores @ v) @ wo
        D = 32
        bc = get_bytecode(
            attention,
            np.ones((8, D), dtype=np.float32),
            np.ones((D, D), dtype=np.float32),
            np.ones((D, D), dtype=np.float32),
            np.ones((D, D), dtype=np.float32),
            np.ones((D, D), dtype=np.float32),
        )
        np.random.seed(42)
        x = np.random.randn(8, D).astype(np.float32) * 0.1
        wq = np.random.randn(D, D).astype(np.float32) * 0.1
        wk = np.random.randn(D, D).astype(np.float32) * 0.1
        wv = np.random.randn(D, D).astype(np.float32) * 0.1
        wo = np.random.randn(D, D).astype(np.float32) * 0.1
        [result] = execute_stablehlo(bc, [x, wq, wk, wv, wo])
        # Numpy reference
        q = x @ wq
        k = x @ wk
        v = x @ wv
        scores = q @ k.T / np.sqrt(D)
        scores_exp = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)
        ref = (attn @ v) @ wo
        np.testing.assert_allclose(result, ref, rtol=1e-4, atol=1e-5)


# ============================================================
# Tests: Function calls (private functions like relu, silu)
# ============================================================

class TestFuncCall:
    def test_mlp_with_relu(self):
        """MLP with relu (relu is a private function)"""
        import jax
        import jax.numpy as jnp
        def mlp(x, w1, b1, w2, b2):
            h = jax.nn.relu(x @ w1 + b1)
            return h @ w2 + b2
        bc = get_bytecode(
            mlp,
            np.ones((2, 32), dtype=np.float32),
            np.ones((32, 64), dtype=np.float32),
            np.ones(64, dtype=np.float32),
            np.ones((64, 32), dtype=np.float32),
            np.ones(32, dtype=np.float32),
        )
        np.random.seed(42)
        x = np.random.randn(2, 32).astype(np.float32) * 0.1
        w1 = np.random.randn(32, 64).astype(np.float32) * 0.1
        b1 = np.zeros(64, dtype=np.float32)
        w2 = np.random.randn(64, 32).astype(np.float32) * 0.1
        b2 = np.zeros(32, dtype=np.float32)
        [result] = execute_stablehlo(bc, [x, w1, b1, w2, b2])
        ref = np.maximum(x @ w1 + b1, 0) @ w2 + b2
        np.testing.assert_allclose(result, ref, rtol=1e-4, atol=1e-5)

    def test_silu_mlp(self):
        """SiLU gated MLP (silu is a private function)"""
        import jax
        import jax.numpy as jnp
        def silu_mlp(x, w_gate, w_up, w_down):
            gate = jax.nn.silu(x @ w_gate)
            up = x @ w_up
            return (gate * up) @ w_down
        bc = get_bytecode(
            silu_mlp,
            np.ones((2, 32), dtype=np.float32),
            np.ones((32, 64), dtype=np.float32),
            np.ones((32, 64), dtype=np.float32),
            np.ones((64, 32), dtype=np.float32),
        )
        np.random.seed(42)
        x = np.random.randn(2, 32).astype(np.float32) * 0.1
        w_gate = np.random.randn(32, 64).astype(np.float32) * 0.1
        w_up = np.random.randn(32, 64).astype(np.float32) * 0.1
        w_down = np.random.randn(64, 32).astype(np.float32) * 0.1
        [result] = execute_stablehlo(bc, [x, w_gate, w_up, w_down])
        # SiLU = x * sigmoid(x) = x / (1 + exp(-x))
        gate_val = x @ w_gate
        silu = gate_val / (1 + np.exp(-gate_val))
        ref = (silu * (x @ w_up)) @ w_down
        np.testing.assert_allclose(result, ref, rtol=1e-4, atol=1e-5)


# ============================================================
# Tests: New ops (slice, compare, select, iota, concatenate, and/or)
# ============================================================

class TestSlice:
    def test_static_slice(self):
        """slice x[:, :, :8, :] from a 4D tensor"""
        import jax.numpy as jnp
        bc = get_bytecode(
            lambda x: x[:, :, :8, :],
            np.ones((1, 4, 32, 16), dtype=np.float32),
        )
        x = np.random.randn(1, 4, 32, 16).astype(np.float32)
        [result] = execute_stablehlo(bc, [x])
        np.testing.assert_allclose(result, x[:, :, :8, :])

    def test_slice_middle(self):
        """slice x[1:3] from a 1D tensor"""
        import jax.numpy as jnp
        bc = get_bytecode(
            lambda x: x[1:3],
            np.ones(8, dtype=np.float32),
        )
        x = np.arange(8, dtype=np.float32)
        [result] = execute_stablehlo(bc, [x])
        np.testing.assert_allclose(result, x[1:3])


class TestCompareSelect:
    def test_where(self):
        """jnp.where(x > 0, x, 0.0)"""
        import jax.numpy as jnp
        bc = get_bytecode(
            lambda x: jnp.where(x > 0, x, 0.0),
            np.ones(8, dtype=np.float32),
        )
        x = np.array([-2, -1, 0, 0.5, 1, 2, -0.5, 3], dtype=np.float32)
        [result] = execute_stablehlo(bc, [x])
        np.testing.assert_allclose(result, np.where(x > 0, x, 0.0))

    def test_compare_gt(self):
        """x > 0.5"""
        import jax.numpy as jnp
        bc = get_bytecode(
            lambda x: x > 0.5,
            np.ones(4, dtype=np.float32),
        )
        x = np.array([0.0, 0.5, 0.6, 1.0], dtype=np.float32)
        [result] = execute_stablehlo(bc, [x])
        np.testing.assert_array_equal(result, x > 0.5)

    def test_tril(self):
        """jnp.tril generates iota + compare + select"""
        import jax.numpy as jnp
        bc = get_bytecode(
            lambda x: jnp.tril(x),
            np.ones((4, 4), dtype=np.float32),
        )
        x = np.ones((4, 4), dtype=np.float32)
        [result] = execute_stablehlo(bc, [x])
        np.testing.assert_allclose(result, np.tril(x))


class TestIota:
    def test_iota_1d(self):
        """jnp.arange generates iota"""
        import jax.numpy as jnp
        bc = get_bytecode(
            lambda x: jnp.arange(x.shape[0]),
            np.ones(8, dtype=np.float32),
        )
        x = np.ones(8, dtype=np.float32)
        [result] = execute_stablehlo(bc, [x])
        np.testing.assert_array_equal(result, np.arange(8))


class TestConcatenate:
    def test_concat_1d(self):
        """concatenate two 1D arrays"""
        import jax.numpy as jnp
        bc = get_bytecode(
            lambda x, y: jnp.concatenate([x, y]),
            np.ones(3, dtype=np.float32),
            np.ones(4, dtype=np.float32),
        )
        x = np.array([1, 2, 3], dtype=np.float32)
        y = np.array([4, 5, 6, 7], dtype=np.float32)
        [result] = execute_stablehlo(bc, [x, y])
        np.testing.assert_allclose(result, np.concatenate([x, y]))

    def test_concat_axis1(self):
        """concatenate along axis=1"""
        import jax.numpy as jnp
        bc = get_bytecode(
            lambda x, y: jnp.concatenate([x, y], axis=-1),
            np.ones((2, 3), dtype=np.float32),
            np.ones((2, 4), dtype=np.float32),
        )
        x = np.random.randn(2, 3).astype(np.float32)
        y = np.random.randn(2, 4).astype(np.float32)
        [result] = execute_stablehlo(bc, [x, y])
        np.testing.assert_allclose(result, np.concatenate([x, y], axis=-1))


class TestBooleanOps:
    def test_and(self):
        """(x > 0) & (y > 0)"""
        import jax.numpy as jnp
        bc = get_bytecode(
            lambda x, y: (x > 0) & (y > 0),
            np.ones(4, dtype=np.float32),
            np.ones(4, dtype=np.float32),
        )
        x = np.array([1, -1, 1, -1], dtype=np.float32)
        y = np.array([1, 1, -1, -1], dtype=np.float32)
        [result] = execute_stablehlo(bc, [x, y])
        np.testing.assert_array_equal(result, (x > 0) & (y > 0))
