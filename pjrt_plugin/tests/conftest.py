"""Shared test fixtures for PJRT plugin tests.

Handles plugin registration and provides a TT device fixture.
All tests in this directory can use the `tt_device` fixture.
"""

import os
import sys
import pytest

# Add the plugin directory to the path so we can import jax_plugins.tt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="session", autouse=True)
def register_plugin():
    """Register the TT PJRT plugin before any tests run."""
    from jax_plugins.tt import initialize

    try:
        initialize()
    except RuntimeError as e:
        pytest.skip(f"Plugin not built: {e}")


@pytest.fixture
def tt_device():
    """Return the first TT device, or skip if not available."""
    import jax

    devices = jax.devices("tt")
    if not devices:
        pytest.skip("No TT devices found")
    return devices[0]
