"""Test the StableHLO Python interpreter engine.

Tests that the engine correctly parses MLIR bytecode and executes
StableHLO ops on numpy arrays. These tests validate the Python engine
independently of the C++ PJRT plugin.

IMPORTANT: These tests use CPU-only JAX. The TT plugin must NOT be loaded
because it would become the default backend and crash on jnp.ones().
"""

import sys
import os

# Force CPU-only mode BEFORE importing JAX (prevents TT plugin auto-load)
os.environ["JAX_PLATFORMS"] = "cpu"

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
    """JIT-lower a function and return its StableHLO bytecode."""
    import jax
    import jax._src.interpreters.mlir as jax_mlir

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
        import jax.numpy as jnp
        from jaxlib.mlir._mlir_libs._stablehlo import (
            serialize_portable_artifact, get_current_version
        )

        f = jax.jit(lambda x: x + 1.0)
        lowered = f.lower(jnp.ones(4))
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
        bc = get_bytecode(lambda x: x + 1.0, jnp.ones(4))
        text = bytecode_to_text(bc)
        assert "stablehlo.add" in text
        assert "stablehlo.constant" in text

    def test_parse_add_program(self):
        import jax.numpy as jnp
        bc = get_bytecode(lambda x: x + 1.0, jnp.ones(4))
        text = bytecode_to_text(bc)
        args, ops, returns = parse_stablehlo(text)

        assert len(args) == 1
        assert args[0][0] == "arg0"
        assert len(returns) >= 1

        op_types = [op['op'] for op in ops]
        assert 'constant' in op_types
        assert 'add' in op_types

    def test_parse_matmul_program(self):
        import jax.numpy as jnp
        bc = get_bytecode(lambda x, w: x @ w, jnp.ones((2, 3)), jnp.ones((3, 4)))
        text = bytecode_to_text(bc)
        args, ops, returns = parse_stablehlo(text)

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
        bc = get_bytecode(lambda x: x + 1.0, jnp.ones(4))
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        [result] = execute_stablehlo(bc, [x])
        np.testing.assert_allclose(result, x + 1.0)

    def test_multiply_add(self):
        """x * 2 + 3"""
        import jax.numpy as jnp
        bc = get_bytecode(lambda x: x * 2.0 + 3.0, jnp.ones(4))
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        [result] = execute_stablehlo(bc, [x])
        np.testing.assert_allclose(result, x * 2.0 + 3.0)

    def test_subtract(self):
        """x - y"""
        import jax.numpy as jnp
        bc = get_bytecode(lambda x, y: x - y, jnp.ones(4), jnp.ones(4))
        x = np.array([5.0, 4.0, 3.0, 2.0], dtype=np.float32)
        y = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        [result] = execute_stablehlo(bc, [x, y])
        np.testing.assert_allclose(result, x - y)

    def test_negate(self):
        """-x"""
        import jax.numpy as jnp
        bc = get_bytecode(lambda x: -x, jnp.ones(4))
        x = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
        [result] = execute_stablehlo(bc, [x])
        np.testing.assert_allclose(result, -x)

    def test_exp(self):
        """exp(x)"""
        import jax.numpy as jnp
        bc = get_bytecode(lambda x: jnp.exp(x), jnp.ones(4))
        x = np.array([0.0, 1.0, 2.0, -1.0], dtype=np.float32)
        [result] = execute_stablehlo(bc, [x])
        np.testing.assert_allclose(result, np.exp(x), rtol=1e-6)

    def test_tanh(self):
        """tanh(x)"""
        import jax.numpy as jnp
        bc = get_bytecode(lambda x: jnp.tanh(x), jnp.ones(4))
        x = np.array([0.0, 1.0, -1.0, 3.0], dtype=np.float32)
        [result] = execute_stablehlo(bc, [x])
        np.testing.assert_allclose(result, np.tanh(x), rtol=1e-6)

    def test_matmul(self):
        """x @ w"""
        import jax.numpy as jnp
        bc = get_bytecode(lambda x, w: x @ w, jnp.ones((2, 3)), jnp.ones((3, 4)))
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        w = np.eye(3, 4, dtype=np.float32) * 2
        [result] = execute_stablehlo(bc, [x, w])
        np.testing.assert_allclose(result, x @ w)

    def test_matmul_add(self):
        """x @ w + b (linear layer)"""
        import jax.numpy as jnp
        bc = get_bytecode(
            lambda x, w, b: x @ w + b,
            jnp.ones((2, 3)), jnp.ones((3, 4)), jnp.ones(4)
        )
        x = np.random.randn(2, 3).astype(np.float32)
        w = np.random.randn(3, 4).astype(np.float32)
        b = np.random.randn(4).astype(np.float32)
        [result] = execute_stablehlo(bc, [x, w, b])
        np.testing.assert_allclose(result, x @ w + b, rtol=1e-5)

    def test_larger_array(self):
        """Larger array round-trip through computation."""
        import jax.numpy as jnp
        bc = get_bytecode(lambda x: x * 3.14 + 2.71, jnp.ones((32, 64)))
        x = np.random.randn(32, 64).astype(np.float32)
        [result] = execute_stablehlo(bc, [x])
        np.testing.assert_allclose(result, x * 3.14 + 2.71, rtol=1e-5)

    def test_element_multiply(self):
        """x * y element-wise"""
        import jax.numpy as jnp
        bc = get_bytecode(lambda x, y: x * y, jnp.ones(4), jnp.ones(4))
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        y = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        [result] = execute_stablehlo(bc, [x, y])
        np.testing.assert_allclose(result, x * y)

    def test_divide(self):
        """x / y"""
        import jax.numpy as jnp
        bc = get_bytecode(lambda x, y: x / y, jnp.ones(4), jnp.ones(4))
        x = np.array([6.0, 8.0, 9.0, 12.0], dtype=np.float32)
        y = np.array([2.0, 4.0, 3.0, 6.0], dtype=np.float32)
        [result] = execute_stablehlo(bc, [x, y])
        np.testing.assert_allclose(result, x / y)

    def test_maximum(self):
        """max(x, 0) (relu)"""
        import jax.numpy as jnp
        bc = get_bytecode(lambda x: jnp.maximum(x, 0.0), jnp.ones(4))
        x = np.array([-2.0, -1.0, 0.0, 1.0], dtype=np.float32)
        [result] = execute_stablehlo(bc, [x])
        np.testing.assert_allclose(result, np.maximum(x, 0.0))
