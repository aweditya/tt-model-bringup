"""Test buffer management (Phase 2).

Tests that data survives host -> device -> host round trips
via BufferFromHostBuffer and ToHostBuffer.
"""

import pytest
import numpy as np


def test_host_to_device_roundtrip(tt_device):
    """Test that data survives a host -> device -> host round trip."""
    import jax
    import jax.numpy as jnp

    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    y = jax.device_put(x, tt_device)
    z = jax.device_get(y)
    np.testing.assert_allclose(z, x, atol=1e-6)


def test_buffer_metadata(tt_device):
    """Test that buffer reports correct shape and dtype."""
    import jax
    import jax.numpy as jnp

    x = np.ones((2, 3), dtype=np.float32)
    buf = jax.device_put(x, tt_device)
    assert buf.shape == (2, 3)
    assert buf.dtype == jnp.float32


def test_scalar_roundtrip(tt_device):
    """Test scalar values."""
    import jax

    x = np.float32(42.0)
    y = jax.device_put(x, tt_device)
    z = jax.device_get(y)
    np.testing.assert_allclose(z, x, atol=1e-6)


def test_1d_roundtrip(tt_device):
    """Test 1D array."""
    import jax

    x = np.arange(10, dtype=np.float32)
    y = jax.device_put(x, tt_device)
    z = jax.device_get(y)
    np.testing.assert_allclose(z, x, atol=1e-6)


def test_large_array_roundtrip(tt_device):
    """Test a larger array to catch any size-related issues."""
    import jax

    x = np.random.randn(64, 128).astype(np.float32)
    y = jax.device_put(x, tt_device)
    z = jax.device_get(y)
    np.testing.assert_allclose(z, x, atol=1e-6)


def test_int32_roundtrip(tt_device):
    """Test integer dtype."""
    import jax

    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = jax.device_put(x, tt_device)
    z = jax.device_get(y)
    np.testing.assert_array_equal(z, x)
