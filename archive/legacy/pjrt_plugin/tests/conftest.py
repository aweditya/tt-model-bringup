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
    """Try to register the TT PJRT plugin. Engine tests don't need it.

    JAX auto-discovers plugins via the jax_plugins namespace package, so
    `import jax` already triggers initialize(). Catch ALREADY_EXISTS so
    the test fixture is idempotent.
    """
    global _plugin_available
    from jax_plugins.tt import initialize

    try:
        initialize()
        _plugin_available = True
    except RuntimeError:
        _plugin_available = False
    except Exception as e:
        # jaxlib's XlaRuntimeError is not a RuntimeError; tolerate it.
        if 'ALREADY_EXISTS' in str(e):
            _plugin_available = True
        else:
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


@pytest.fixture
def tols():
    """Numerical tolerances that adapt to execution mode.

    Returns (atol, rtol) tuple. In device mode (TT_PJRT_USE_DEVICE=1) the
    engine runs in bf16, which has ~7 bits of mantissa and visibly drifts
    on transcendentals, matmuls, and reductions. In numpy mode it runs in
    fp32 and we expect much tighter agreement.
    """
    if os.environ.get('TT_PJRT_USE_DEVICE', '0') == '1':
        return {'atol': 1e-2, 'rtol': 5e-2}
    return {'atol': 1e-5, 'rtol': 1e-4}
