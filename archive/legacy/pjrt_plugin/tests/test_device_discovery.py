"""Phase 1 test: verify JAX can discover our TT device.

This is the first test that should pass. It verifies:
1. Plugin loads successfully via dlopen/GetPjrtApi
2. Client creates without error (ttnn device opens)
3. jax.devices() shows our Blackhole device
4. Device metadata (kind, id) is correct

Run with: python -m pytest tests/test_device_discovery.py -v
"""

import pytest


def test_device_visible(tt_device):
    """jax.devices('tt') should return at least one device."""
    assert tt_device is not None


def test_device_platform():
    """Platform name should be 'tt'."""
    import jax

    devices = jax.devices("tt")
    assert len(devices) >= 1
    # JAX exposes platform via the device's platform attribute
    assert devices[0].platform == "tt"


def test_device_id(tt_device):
    """Device 0 should have id 0."""
    assert tt_device.id == 0


def test_device_kind(tt_device):
    """Device kind should be 'Blackhole'."""
    # JAX may expose this differently depending on version.
    # The device_kind is set in DeviceDescription_Kind.
    assert "Blackhole" in str(tt_device) or tt_device.device_kind == "Blackhole"


def test_single_device():
    """We should have exactly one TT device (device 0 only)."""
    import jax

    devices = jax.devices("tt")
    assert len(devices) == 1
