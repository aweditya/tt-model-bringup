"""Test dot_general / matmul (Phase 3).

Dedicated matmul tests covering batched, transposed, and non-square cases.
"""

import pytest
import numpy as np


@pytest.mark.skip(reason="Phase 3: execution not yet implemented")
def test_dot_general_basic(tt_device):
    """Test basic matrix multiply via dot_general."""
    import jax
    import jax.numpy as jnp

    @jax.jit
    def linear(x, w, b):
        return x @ w + b

    x = jnp.ones((2, 4))
    w = jnp.ones((4, 4))
    b = jnp.ones((4,))
    y = linear(x, w, b)
    # Each row: 4 * 1.0 + 1.0 = 5.0
    np.testing.assert_allclose(jax.device_get(y), 5.0 * np.ones((2, 4)), atol=1e-2)


@pytest.mark.skip(reason="Phase 3: execution not yet implemented")
def test_dot_general_non_square(tt_device):
    """Test non-square matrix multiply."""
    import jax
    import jax.numpy as jnp

    @jax.jit
    def matmul(a, b):
        return a @ b

    a = jnp.ones((3, 5))
    b = jnp.ones((5, 7))
    c = matmul(a, b)
    np.testing.assert_allclose(jax.device_get(c), 5.0 * np.ones((3, 7)), atol=1e-2)
