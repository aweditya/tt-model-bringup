"""Shared test fixtures for PJRT plugin tests.

Handles plugin registration and provides a TT device fixture.
All tests in this directory can use the `tt_device` fixture.
"""

import os
import sys
import pytest

# Add the plugin directory to the path so we can import jax_plugins.tt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


_plugin_available = False

@pytest.fixture(scope="session", autouse=True)
def register_plugin():
    """Try to register the TT PJRT plugin. Engine tests don't need it."""
    global _plugin_available
    from jax_plugins.tt import initialize

    try:
        initialize()
        _plugin_available = True
    except RuntimeError:
        _plugin_available = False


@pytest.fixture
def tt_device():
    """Return the first TT device, or skip if not available."""
    if not _plugin_available:
        pytest.skip("TT plugin not built")
    import jax

    devices = jax.devices("tt")
    if not devices:
        pytest.skip("No TT devices found")
    return devices[0]
