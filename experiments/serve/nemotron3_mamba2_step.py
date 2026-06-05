"""MM7 G4 — Python wrapper for the owned Mamba2 SSD decode kernel.

Bridges the kernel's per-(batch, head)-tile contract to the
host-facing decode step API that the upcoming
`server_nemotron3_ttnn.py` will call once per Mamba2 layer per token.

Contract (matches the numpy oracle `mamba2_decode_step` from
`experiments/utils/mamba2_numpy_oracle.py`):

    new_state, y = mamba2_decode_step(
        x:         np.ndarray | torch.Tensor,   # (B, NUM_HEADS, HEAD_DIM)
        z:         np.ndarray | torch.Tensor,   # (B, NUM_HEADS, HEAD_DIM)  pass-through
        dt:        np.ndarray | torch.Tensor,   # (B, NUM_HEADS)
        dt_bias:   np.ndarray | torch.Tensor,   # (NUM_HEADS,)
        A_log:     np.ndarray | torch.Tensor,   # (NUM_HEADS,)
        D:         np.ndarray | torch.Tensor,   # (NUM_HEADS,)
        B_in:      np.ndarray | torch.Tensor,   # (B, N_GROUPS, SSM_STATE)
        C_in:      np.ndarray | torch.Tensor,   # (B, N_GROUPS, SSM_STATE)
        ssm_state: np.ndarray | torch.Tensor,   # (B, NUM_HEADS, HEAD_DIM, SSM_STATE)  fp32
        *,
        device,                                  # open ttnn device handle
        debug_mode: int = 5,                     # production
    ) -> (new_state, y)
        new_state: same shape + dtype as ssm_state
        y:         same shape + dtype as x

Internally:
  1. Reshape inputs to the kernel's per-block tile layout:
      - x/z/dt: pad per-head plane to 32 rows (values in row 0)
      - dt_bias/A_log/D: pad to (B, NUM_HEADS, 32, 32) with value at [0,0]
      - B_in/C_in: replicate per-group → per-head, pad to (B, NUM_HEADS, 32, SSM_STATE)
      - ssm_state: pass through (already (B, NUM_HEADS, HEAD_DIM, SSM_STATE) fp32)
  2. Upload as ttnn tensors (bf16 for the inputs, fp32 for state)
  3. Invoke `ttnn.experimental.nemotron3_mamba2_decode_owned(..., debug_mode=5)`
  4. Read back state + y, slice off the row-0 padding, return numpy.

This wrapper is single-step; the CB engine + server loop iterate it
per token. For the G2 + G3 path (multi-core / multi-batch), see the
program_factory — the kernel handles those automatically once shapes
are right.

REUSE: bundle the host-side padding helpers from
`experiments/cb/isolate/mamba2_g2_multihead_smoke.py` (G2 smoke) into
a single reusable module so the server scaffold and any future probes
share one source of truth.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

# We import ttnn lazily inside the call so unit tests that don't need
# the device path can `import nemotron3_mamba2_step` without ttnn installed.


def _to_numpy(arr) -> np.ndarray:
    if isinstance(arr, torch.Tensor):
        return arr.detach().float().cpu().numpy()
    return np.asarray(arr)


def _pad_per_head_vector(arr: np.ndarray, last_dim: int) -> np.ndarray:
    """(B, NUM_HEADS, last_dim) → (B, NUM_HEADS, 32, last_dim).
    Values land in row 0 of each per-head (32, last_dim) plane.
    """
    B, H, D = arr.shape
    assert D == last_dim, f"last dim {D} ≠ expected {last_dim}"
    out = np.zeros((B, H, 32, last_dim), dtype=np.float32)
    out[:, :, 0, :] = arr
    return out


def _pad_scalar_per_head_per_batch(arr: np.ndarray, B: int, num_heads: int) -> np.ndarray:
    """(NUM_HEADS,) → (B, NUM_HEADS, 32, 32).
    Each (h, 32, 32) tile has its scalar at cell [0, 0]. Replicated
    across the batch axis so the reader's global_block tile-index
    points at the right head for any (b, h) block.
    """
    if arr.ndim != 1 or arr.shape[0] != num_heads:
        raise ValueError(
            f"expected per-head scalar shape ({num_heads},), got {arr.shape}")
    out = np.zeros((B, num_heads, 32, 32), dtype=np.float32)
    out[..., 0, 0] = arr[None, :]
    return out


def _pad_dt_per_batch_per_head(arr: np.ndarray) -> np.ndarray:
    """(B, NUM_HEADS) → (B, NUM_HEADS, 32, 32) with values at [b, h, 0, 0]."""
    B, H = arr.shape
    out = np.zeros((B, H, 32, 32), dtype=np.float32)
    out[..., 0, 0] = arr
    return out


def _replicate_per_group_to_per_head(
    group_arr: np.ndarray, num_heads: int
) -> np.ndarray:
    """(B, N_GROUPS, SSM_STATE) → (B, NUM_HEADS, 32, SSM_STATE).

    Each head's row-0 plane carries its group's B (or C) vector.
    NUM_HEADS must be divisible by N_GROUPS.
    """
    B, G, D = group_arr.shape
    if num_heads % G != 0:
        raise ValueError(
            f"num_heads {num_heads} not divisible by n_groups {G}")
    heads_per_group = num_heads // G
    out = np.zeros((B, num_heads, 32, D), dtype=np.float32)
    for h in range(num_heads):
        g = h // heads_per_group
        out[:, h, 0, :] = group_arr[:, g, :]
    return out


def mamba2_decode_step_ttnn(
    *,
    x,
    z,
    dt,
    dt_bias,
    A_log,
    D,
    B_in,
    C_in,
    ssm_state,
    device,
    debug_mode: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Single-step Mamba2 SSD decode, via the owned kernel.

    All inputs are numpy or torch tensors at logical shapes (see module
    docstring). Returns (new_state, y) as numpy arrays. The input
    `ssm_state` is NOT mutated; callers who want the threaded-state
    contract of `mamba2_numpy_oracle.mamba2_decode_step` should
    overwrite their state buffer with the returned `new_state`.
    """
    import ttnn  # lazy import — see module docstring

    # ── Normalise to numpy fp32 ──────────────────────────────────────
    x_np         = _to_numpy(x).astype(np.float32)
    z_np         = _to_numpy(z).astype(np.float32)
    dt_np        = _to_numpy(dt).astype(np.float32)
    dt_bias_np   = _to_numpy(dt_bias).astype(np.float32)
    A_log_np     = _to_numpy(A_log).astype(np.float32)
    D_np         = _to_numpy(D).astype(np.float32)
    B_in_np      = _to_numpy(B_in).astype(np.float32)
    C_in_np      = _to_numpy(C_in).astype(np.float32)
    ssm_state_np = _to_numpy(ssm_state).astype(np.float32)

    B_, num_heads, head_dim = x_np.shape
    ssm_state_size = B_in_np.shape[-1]

    # ── Pad to per-block tile layout the reader expects ────────────────
    x_padded   = _pad_per_head_vector(x_np, head_dim)
    z_padded   = _pad_per_head_vector(z_np, head_dim)
    dt_padded  = _pad_dt_per_batch_per_head(dt_np)
    dt_bias_padded = _pad_scalar_per_head_per_batch(dt_bias_np, B_, num_heads)
    A_log_padded   = _pad_scalar_per_head_per_batch(A_log_np, B_, num_heads)
    D_padded       = _pad_scalar_per_head_per_batch(D_np, B_, num_heads)
    B_padded = _replicate_per_group_to_per_head(B_in_np, num_heads)
    C_padded = _replicate_per_group_to_per_head(C_in_np, num_heads)

    # ── Upload ──────────────────────────────────────────────────────
    # Detect mesh device — needs ReplicateTensorToMesh on upload AND
    # ConcatMeshToTensor + [:1] on readback. Single-device path stays
    # as-is. The Nemotron-3 server uses a (1,4) mesh; the original G2/G3
    # smoke probes used a single device.
    is_mesh = isinstance(device, ttnn.MeshDevice)

    def tt_bf16(arr):
        kwargs = dict(
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
        )
        if is_mesh:
            kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(device)
        return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(arr)),
                                 **kwargs)

    def tt_fp32(arr):
        kwargs = dict(
            dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device,
        )
        if is_mesh:
            kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(device)
        return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(arr)),
                                 **kwargs)

    x_tt        = tt_bf16(x_padded)
    z_tt        = tt_bf16(z_padded)
    dt_tt       = tt_bf16(dt_padded)
    dt_bias_tt  = tt_bf16(dt_bias_padded)
    A_log_tt    = tt_bf16(A_log_padded)
    D_tt        = tt_bf16(D_padded)
    B_in_tt     = tt_bf16(B_padded)
    C_in_tt     = tt_bf16(C_padded)
    ssm_state_tt = tt_fp32(ssm_state_np)

    # ── Invoke kernel ────────────────────────────────────────────────
    state_out_tt, y_out_tt = ttnn.experimental.nemotron3_mamba2_decode_owned(
        x_tt, z_tt, dt_tt, dt_bias_tt, A_log_tt, D_tt,
        B_in_tt, C_in_tt, ssm_state_tt,
        debug_mode=debug_mode,
    )

    # ── Read back (mesh-aware) ───────────────────────────────────────
    if is_mesh:
        composer = ttnn.ConcatMeshToTensor(device, dim=0)
        state_kernel = ttnn.to_torch(state_out_tt, mesh_composer=composer)
        # Replicated → 4 identical chip slabs along dim 0; keep chip 0.
        state_kernel = state_kernel[:B_].cpu().numpy()
        y_kernel_padded = ttnn.to_torch(
            ttnn.typecast(y_out_tt, ttnn.float32),
            mesh_composer=composer)
        y_kernel_padded = y_kernel_padded[:B_].cpu().numpy()
    else:
        state_kernel = ttnn.to_torch(state_out_tt).cpu().numpy()
        y_kernel_padded = ttnn.to_torch(
            ttnn.typecast(y_out_tt, ttnn.float32)).cpu().numpy()

    # y came back at (B, NUM_HEADS, 32, HEAD_DIM); extract row 0.
    if y_kernel_padded.ndim == 4 and y_kernel_padded.shape[-2] == 32:
        y_logical = y_kernel_padded[:, :, 0, :head_dim]
    else:
        y_logical = y_kernel_padded[:, :, :head_dim].reshape(
            B_, num_heads, head_dim)

    state_logical = state_kernel[:, :, :head_dim, :ssm_state_size].reshape(
        B_, num_heads, head_dim, ssm_state_size)

    return state_logical, y_logical


__all__ = ["mamba2_decode_step_ttnn"]
