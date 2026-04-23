"""Test basic arithmetic ops (Phase 3).

These tests will work once Compile and Execute are implemented.
For now they are expected to fail with UNIMPLEMENTED.
"""

import pytest
import numpy as np


@pytest.mark.skip(reason="Phase 3: execution not yet implemented")
def test_add(tt_device):
    """Test jax.jit addition on TT device."""
    import jax
    import jax.numpy as jnp

    @jax.jit
    def add(a, b):
        return a + b

    a = jnp.ones((4, 4))
    b = jnp.ones((4, 4))
    c = add(a, b)
    np.testing.assert_allclose(jax.device_get(c), 2.0 * np.ones((4, 4)), atol=1e-2)


@pytest.mark.skip(reason="Phase 3: execution not yet implemented")
def test_matmul(tt_device):
    """Test jax.jit matrix multiplication on TT device."""
    import jax
    import jax.numpy as jnp

    @jax.jit
    def matmul(a, b):
        return a @ b

    a = jnp.ones((4, 4))
    b = jnp.ones((4, 4))
    c = matmul(a, b)
    expected = 4.0 * np.ones((4, 4))
    np.testing.assert_allclose(jax.device_get(c), expected, atol=1e-2)
