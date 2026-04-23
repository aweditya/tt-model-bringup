"""Test basic JAX operations through the full PJRT pipeline (Phase 3).

These tests verify that jax.jit(f)(x) works end-to-end on the TT backend:
JAX → StableHLO → PJRT Compile → PJRT Execute → result.
"""

import pytest
import numpy as np


class TestArithmetic:
    def test_add_scalar(self, tt_device):
        """jax.jit(lambda x: x + 1)(x)"""
        import jax
        f = jax.jit(lambda x: x + 1.0)
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = f(jax.device_put(x, tt_device))
        result = jax.device_get(result)
        np.testing.assert_allclose(result, x + 1.0, atol=1e-6)

    def test_multiply_add(self, tt_device):
        """jax.jit(lambda x: x * 2 + 3)(x)"""
        import jax
        f = jax.jit(lambda x: x * 2.0 + 3.0)
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = jax.device_get(f(jax.device_put(x, tt_device)))
        np.testing.assert_allclose(result, x * 2.0 + 3.0, atol=1e-6)

    def test_subtract(self, tt_device):
        """jax.jit(lambda x, y: x - y)(x, y)"""
        import jax
        f = jax.jit(lambda x, y: x - y)
        x = np.array([5.0, 4.0, 3.0, 2.0], dtype=np.float32)
        y = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        result = jax.device_get(f(
            jax.device_put(x, tt_device),
            jax.device_put(y, tt_device),
        ))
        np.testing.assert_allclose(result, x - y, atol=1e-6)

    def test_divide(self, tt_device):
        """jax.jit(lambda x, y: x / y)(x, y)"""
        import jax
        f = jax.jit(lambda x, y: x / y)
        x = np.array([6.0, 8.0, 9.0, 12.0], dtype=np.float32)
        y = np.array([2.0, 4.0, 3.0, 6.0], dtype=np.float32)
        result = jax.device_get(f(
            jax.device_put(x, tt_device),
            jax.device_put(y, tt_device),
        ))
        np.testing.assert_allclose(result, x / y, atol=1e-6)


class TestUnaryOps:
    def test_negate(self, tt_device):
        import jax
        f = jax.jit(lambda x: -x)
        x = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
        result = jax.device_get(f(jax.device_put(x, tt_device)))
        np.testing.assert_allclose(result, -x, atol=1e-6)

    def test_exp(self, tt_device):
        import jax
        import jax.numpy as jnp
        f = jax.jit(lambda x: jnp.exp(x))
        x = np.array([0.0, 1.0, 2.0, -1.0], dtype=np.float32)
        result = jax.device_get(f(jax.device_put(x, tt_device)))
        np.testing.assert_allclose(result, np.exp(x), rtol=1e-5)

    def test_tanh(self, tt_device):
        import jax
        import jax.numpy as jnp
        f = jax.jit(lambda x: jnp.tanh(x))
        x = np.array([0.0, 1.0, -1.0, 3.0], dtype=np.float32)
        result = jax.device_get(f(jax.device_put(x, tt_device)))
        np.testing.assert_allclose(result, np.tanh(x), rtol=1e-5)

    def test_relu(self, tt_device):
        """max(x, 0)"""
        import jax
        import jax.numpy as jnp
        f = jax.jit(lambda x: jnp.maximum(x, 0.0))
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        result = jax.device_get(f(jax.device_put(x, tt_device)))
        np.testing.assert_allclose(result, np.maximum(x, 0.0), atol=1e-6)


class TestMatmul:
    def test_simple_matmul(self, tt_device):
        """x @ w"""
        import jax
        f = jax.jit(lambda x, w: x @ w)
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        w = np.eye(3, 4, dtype=np.float32) * 2
        result = jax.device_get(f(
            jax.device_put(x, tt_device),
            jax.device_put(w, tt_device),
        ))
        np.testing.assert_allclose(result, x @ w, atol=1e-5)

    def test_linear_layer(self, tt_device):
        """x @ w + b"""
        import jax
        f = jax.jit(lambda x, w, b: x @ w + b)
        x = np.random.randn(2, 3).astype(np.float32)
        w = np.random.randn(3, 4).astype(np.float32)
        b = np.random.randn(4).astype(np.float32)
        result = jax.device_get(f(
            jax.device_put(x, tt_device),
            jax.device_put(w, tt_device),
            jax.device_put(b, tt_device),
        ))
        np.testing.assert_allclose(result, x @ w + b, rtol=1e-5)

    def test_larger_matmul(self, tt_device):
        """64x128 @ 128x32"""
        import jax
        f = jax.jit(lambda x, w: x @ w)
        x = np.random.randn(64, 128).astype(np.float32)
        w = np.random.randn(128, 32).astype(np.float32)
        result = jax.device_get(f(
            jax.device_put(x, tt_device),
            jax.device_put(w, tt_device),
        ))
        np.testing.assert_allclose(result, x @ w, rtol=1e-4)
