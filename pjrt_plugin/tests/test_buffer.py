"""Test buffer management (Phase 2).

These tests will work once BufferFromHostBuffer and ToHostBuffer are
implemented. For now they are expected to fail with UNIMPLEMENTED.
"""

import pytest
import numpy as np


@pytest.mark.skip(reason="Phase 2: buffer transfer not yet implemented")
def test_host_to_device_roundtrip(tt_device):
    """Test that data survives a host -> device -> host round trip."""
    import jax
    import jax.numpy as jnp

    x = jnp.ones((4, 4), dtype=jnp.float32)
    y = jax.device_put(x, tt_device)
    z = jax.device_get(y)
    np.testing.assert_allclose(z, x, atol=1e-2)  # bfloat16 precision


@pytest.mark.skip(reason="Phase 2: buffer transfer not yet implemented")
def test_buffer_metadata(tt_device):
    """Test that buffer reports correct shape and dtype."""
    import jax
    import jax.numpy as jnp

    x = jnp.ones((2, 3), dtype=jnp.float32)
    buf = jax.device_put(x, tt_device)
    assert buf.shape == (2, 3)
    assert buf.dtype == jnp.float32
