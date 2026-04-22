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
    ttnn.from_torch handles tile padding internally, preserving the
    logical shape — critical for correct on-device broadcast via repeat.
    """
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

    while val.dim() < 2:
        val = val.unsqueeze(0)

    return ttnn.from_torch(val, dtype=ttnn.bfloat16, device=device,
                           layout=ttnn.TILE_LAYOUT)


def from_device(tensor, shape):
    """Convert a TT-NN tensor back to numpy with correct shape.

    ttnn.to_torch returns the logical shape (no tile padding to remove).
    We just need to reshape to match the expected JAX output shape.
    """
    t = ttnn.to_torch(tensor).float()

    if len(shape) == 0:
        return t.numpy().flatten()[0]

    # Reshape to target — handles rank mismatches from 2D minimum
    try:
        return t.reshape(shape).numpy()
    except RuntimeError:
        return t.squeeze().numpy().reshape(shape)


def is_literal(var):
    """Check if a Jaxpr variable is a compile-time literal."""
    return isinstance(var, jax_core.Literal)


def literal_val(var):
    """Extract the Python value from a Jaxpr literal."""
    return float(var.val)


def broadcast_to_match(a_tt, b_tt, a_shape, b_shape, out_shape, device):
    """Explicitly broadcast tensors to match output shape.

    Uses ttnn.repeat for on-device broadcast when possible, falling back
    to CPU round-trip only when repeat can't handle the shape transformation.

    Returns (a_tt, b_tt) with matching shapes.
    """
    def needs_broadcast(in_s, out_s):
        if len(in_s) != len(out_s):
            return True
        return any(i != o for i, o in zip(in_s, out_s))

    if needs_broadcast(a_shape, out_shape):
        a_tt = _broadcast_tensor(a_tt, a_shape, out_shape, device)

    if needs_broadcast(b_shape, out_shape):
        b_tt = _broadcast_tensor(b_tt, b_shape, out_shape, device)

    return a_tt, b_tt


def _broadcast_tensor(t_tt, in_shape, out_shape, device):
    """Broadcast a tensor from in_shape to out_shape, on-device when possible."""
    # Try on-device repeat first
    try:
        return _repeat_broadcast(t_tt, in_shape, out_shape)
    except Exception:
        pass

    # Fallback: CPU round-trip
    t_np = from_device(t_tt, in_shape)
    t_np = np.broadcast_to(np.array(t_np).reshape(in_shape), out_shape).copy()
    return to_device(t_np, device)


def _repeat_broadcast(t_tt, in_shape, out_shape):
    """Use ttnn.repeat to broadcast on-device.

    ttnn.repeat takes a shape where each dim is the repeat count.
    E.g., (1, 64) -> (32, 64) needs repeat shape (32, 1).

    Since ttnn preserves logical shapes (tile padding is internal),
    we compute repeat counts directly from logical shapes.
    """
    import ttnn as _ttnn

    in_padded = list(in_shape)
    while len(in_padded) < len(out_shape):
        in_padded.insert(0, 1)

    repeat_counts = []
    for i_dim, o_dim in zip(in_padded, out_shape):
        if i_dim == o_dim:
            repeat_counts.append(1)
        elif i_dim == 1:
            repeat_counts.append(o_dim)
        else:
            raise ValueError(f"Cannot repeat {in_shape} to {out_shape}")

    if all(r == 1 for r in repeat_counts):
        return t_tt

    # Pad to match device tensor rank (ttnn may add batch dims)
    dev_rank = len(t_tt.shape)
    while len(repeat_counts) < dev_rank:
        repeat_counts.insert(0, 1)

    return _ttnn.repeat(t_tt, _ttnn.Shape(repeat_counts))
