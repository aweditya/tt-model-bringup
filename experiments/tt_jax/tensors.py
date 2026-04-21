"""
Tensor utilities for host ↔ device transfer with tile alignment.

TT-NN's TILE_LAYOUT requires dimensions to be multiples of 32.
This module handles padding on upload and unpadding on download.
"""

import numpy as np
import torch
import ttnn
from jax._src import core as jax_core


def to_device(val, device):
    """Convert a host value to a TT-NN tensor on device.

    Handles: int, float, numpy array, JAX array, torch tensor.
    Pads to 32-aligned dimensions for TILE_LAYOUT.
    """
    # Convert to torch tensor
    if isinstance(val, (int, float)):
        val = np.array([[val]], dtype=np.float32)
    if isinstance(val, np.ndarray):
        val = torch.from_numpy(val.copy()).float()
    try:
        import jax
        if isinstance(val, jax.Array):
            val = torch.from_numpy(np.array(val)).float()
    except ImportError:
        pass

    # Ensure at least 2D (TILE_LAYOUT requirement)
    while val.dim() < 2:
        val = val.unsqueeze(0)

    # Pad to tile-aligned (multiples of 32)
    h, w = val.shape[-2], val.shape[-1]
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    if pad_h > 0 or pad_w > 0:
        val = torch.nn.functional.pad(val, (0, pad_w, 0, pad_h))

    return ttnn.from_torch(val, dtype=ttnn.bfloat16, device=device,
                           layout=ttnn.TILE_LAYOUT)


def from_device(tensor, shape):
    """Convert a TT-NN tensor back to numpy with correct shape.

    Removes tile padding and restores the original shape.
    """
    t = ttnn.to_torch(tensor).float()

    if len(shape) == 0:
        return t.numpy().flatten()[0]
    elif len(shape) == 1:
        return t.reshape(-1).numpy()[:shape[0]]
    elif len(shape) == 2:
        while t.dim() > 2:
            t = t.squeeze(0)
        return t.numpy()[:shape[0], :shape[1]]
    else:
        return t.squeeze().numpy()


def is_literal(var):
    """Check if a Jaxpr variable is a compile-time literal."""
    return isinstance(var, jax_core.Literal)


def literal_val(var):
    """Extract the Python value from a Jaxpr literal."""
    return float(var.val)


def broadcast_to_match(a_tt, b_tt, a_shape, b_shape, out_shape, device):
    """Explicitly broadcast tensors to match output shape.

    TT-NN TILE_LAYOUT cannot broadcast mismatched shapes (e.g. (32,1) vs (32,64)).
    We round-trip through CPU to broadcast when needed.

    Returns (a_tt, b_tt) with matching shapes.
    """
    def needs_broadcast(in_s, out_s):
        if len(in_s) != len(out_s):
            return True
        return any(i != o for i, o in zip(in_s, out_s))

    if needs_broadcast(a_shape, out_shape):
        a_np = from_device(a_tt, a_shape)
        a_np = np.broadcast_to(np.array(a_np).reshape(a_shape), out_shape).copy()
        a_tt = to_device(a_np, device)

    if needs_broadcast(b_shape, out_shape):
        b_np = from_device(b_tt, b_shape)
        b_np = np.broadcast_to(np.array(b_np).reshape(b_shape), out_shape).copy()
        b_tt = to_device(b_np, device)

    return a_tt, b_tt
