"""Mamba2 SSD decode-step numpy oracle for Nemotron-3 Nano.

Pure-numpy reference for the per-token SSD recursion that the owned
Blackhole kernel will implement (G1..G4 of the MM7 bringup plan, see
`research/nemotron3_nano_30b_a3b_bringup_plan.md` §3a). The kernel
boundary matches `NemotronHMamba2Mixer.forward` POST-conv1d / silu /
split and PRE-MambaRMSNormGated / out_proj:

    y, ssm_state' = mamba2_decode_step(
        x        : [B, num_heads, head_dim]                bf16/fp32
        z        : [B, num_heads, head_dim]                bf16/fp32  (passed through; gate consumed downstream)
        dt       : [B, num_heads]                          bf16/fp32
        B_in     : [B, n_groups, ssm_state]                bf16/fp32  (per-group; broadcast to heads inside)
        C_in     : [B, n_groups, ssm_state]                bf16/fp32
        ssm_state: [B, num_heads, head_dim, ssm_state]     fp32       (mutated in place)
        A_log    : [num_heads]                             bf16/fp32  learned (A = -exp(A_log))
        dt_bias  : [num_heads]                             bf16/fp32  learned
        D        : [num_heads]                             bf16/fp32  learned
    )  -> y : [B, num_heads, head_dim]                     fp32

Math (per the wiki primer `wiki/65_mamba_state_space_models.md` §3
and the architecture brief `research/nemotron3_nano_architecture_brief.md`
§4.3). For each (batch b, head h, channel d, state s):

    dt_eff[b, h] = softplus(dt[b, h] + dt_bias[h])
    dt_eff[b, h] = clamp(dt_eff[b, h], time_step_floor, time_step_max)
    A[h]         = -exp(A_log[h])                          # scalar per head, A < 0
    decay[b, h]  = exp(dt_eff[b, h] * A[h])                # scalar in (0, 1]
    g            = h // (num_heads // n_groups)            # group index for B/C broadcast

    ssm_state'[b, h, d, s] = decay[b, h] * ssm_state[b, h, d, s]
                           + dt_eff[b, h] * B_in[b, g, s] * x[b, h, d]

    y[b, h, d]    = sum_s(C_in[b, g, s] * ssm_state'[b, h, d, s])
                  + D[h] * x[b, h, d]

All math executed in float32 for numerical stability (the on-device
kernel will accumulate in fp32 too — see [[35b-dn-h-state-drift-lever]]
for why bf16 accumulation in a recurrent state drifts). Inputs may be
bf16 (via `ml_dtypes.bfloat16`) or fp32; the function up-casts.

Usage as a library (intended use, for G0a isolation harness):

    from experiments.utils.mamba2_numpy_oracle import mamba2_decode_step
    y = mamba2_decode_step(x, z, dt, B, C, ssm_state, A_log, dt_bias, D)

Usage as a self-test (this file's `__main__`):

    python3 experiments/utils/mamba2_numpy_oracle.py

Prints a deterministic-random fixture's input shapes, runs one decode
step at Nemotron-3 Nano's shapes (num_heads=64, head_dim=64,
ssm_state=128, n_groups=8), and reports basic sanity (state mutated,
output finite, decay in (0, 1], dt_eff in clamp range).

NOTE: This is the SSD recursion only. The `conv1d_step` (with rolling
[B, conv_dim, 4] state) and the `MambaRMSNormGated` + `out_proj` are
SEPARATE pieces that live OUTSIDE the kernel boundary. The conv1d
step gets its own probe (`experiments/cb/isolate/mamba2_conv1d_step.py`,
to be written at G0a); the norm-gated + out_proj are plain ttnn ops.
"""
from __future__ import annotations

import numpy as np

# Default time-step clamp values for Nemotron-3 Nano.
#
# IMPORTANT (oracle drift root cause, v0.1.2.d 2026-06-05): HF Nemotron
# uses `self.time_step_limit = (0.0, inf)` for the clamp in torch_forward
# (modeling_nemotron_h.py: `torch.clamp(dt, self.time_step_limit[0],
# self.time_step_limit[1])`). The config ALSO ships `time_step_min` and
# `time_step_max` fields (0.001, 0.1 in Nemotron) but those are NAMES,
# NOT what HF actually clamps against. We previously hardcoded 0.1 as
# the max from misreading the config, which gave dt_eff values ~2.77x
# smaller than HF — visible as cos=0.943 oracle vs HF y_pre_norm.
DEFAULT_TIME_STEP_FLOOR: float = 0.0
DEFAULT_TIME_STEP_MAX: float = float("inf")


def _stable_softplus(x: np.ndarray) -> np.ndarray:
    """Numerically-stable softplus: log(1 + exp(x)). Avoids overflow
    for large positive x and precision loss for large negative x.
    Equivalent to ``np.logaddexp(0, x)`` but written explicitly so the
    reader can see the math is the same as torch's `F.softplus`.
    """
    return np.logaddexp(np.zeros_like(x), x)


def mamba2_decode_step(
    x: np.ndarray,
    z: np.ndarray,
    dt: np.ndarray,
    B_in: np.ndarray,
    C_in: np.ndarray,
    ssm_state: np.ndarray,
    A_log: np.ndarray,
    dt_bias: np.ndarray,
    D: np.ndarray,
    *,
    time_step_floor: float = DEFAULT_TIME_STEP_FLOOR,
    time_step_max: float = DEFAULT_TIME_STEP_MAX,
) -> np.ndarray:
    """One decode step of Mamba2 SSD. Mutates `ssm_state` in place.

    Args:
        x:         input projection to head channels, shape [B, num_heads, head_dim].
        z:         gate (NOT consumed here; included for kernel signature parity).
        dt:        raw per-head time-step, shape [B, num_heads].
        B_in:      per-group B matrix, shape [B, n_groups, ssm_state]. Broadcast over heads.
        C_in:      per-group C matrix, shape [B, n_groups, ssm_state]. Broadcast over heads.
        ssm_state: persistent SSM state, shape [B, num_heads, head_dim, ssm_state], fp32.
                   MUTATED IN PLACE.
        A_log:     learned `log(-A)` per head, shape [num_heads]. We use A = -exp(A_log).
        dt_bias:   learned per-head bias for the softplus, shape [num_heads].
        D:         learned skip-connection scalar per head, shape [num_heads].
        time_step_floor, time_step_max: clamp limits for dt_eff (defaults match config).

    Returns:
        y: output of the SSD step, shape [B, num_heads, head_dim], fp32. This is
           the pre-norm-gated, pre-out_proj output; downstream code applies
           MambaRMSNormGated(y, z) then out_proj.
    """
    # Shape pre-conditions
    Bsz, num_heads, head_dim = x.shape
    assert z.shape == x.shape, f"z shape {z.shape} must match x shape {x.shape}"
    assert dt.shape == (Bsz, num_heads), f"dt shape {dt.shape} != ({Bsz}, {num_heads})"
    assert ssm_state.shape == (Bsz, num_heads, head_dim, B_in.shape[-1]), (
        f"ssm_state shape {ssm_state.shape} != "
        f"({Bsz}, {num_heads}, {head_dim}, {B_in.shape[-1]})"
    )
    assert A_log.shape == (num_heads,), f"A_log shape {A_log.shape} != ({num_heads},)"
    assert dt_bias.shape == (num_heads,), f"dt_bias shape {dt_bias.shape} != ({num_heads},)"
    assert D.shape == (num_heads,), f"D shape {D.shape} != ({num_heads},)"
    n_groups = B_in.shape[1]
    assert C_in.shape == B_in.shape, f"C_in shape {C_in.shape} != B_in shape {B_in.shape}"
    assert num_heads % n_groups == 0, (
        f"num_heads {num_heads} must be divisible by n_groups {n_groups}"
    )
    heads_per_group = num_heads // n_groups
    ssm_state_dim = ssm_state.shape[-1]
    assert ssm_state.dtype == np.float32, (
        f"ssm_state must be fp32 (got {ssm_state.dtype}); recurrent state in bf16 drifts "
        f"(see [[35b-dn-h-state-drift-lever]])"
    )

    # Up-cast all per-step inputs to fp32 for the math. The on-device kernel will
    # do the analogous up-cast inside its compute engine (HiFi4 / fp32_dest_acc).
    x32 = x.astype(np.float32)
    dt32 = dt.astype(np.float32)
    A_log32 = A_log.astype(np.float32)
    dt_bias32 = dt_bias.astype(np.float32)
    B32 = B_in.astype(np.float32)
    C32 = C_in.astype(np.float32)
    D32 = D.astype(np.float32)

    # Per-(batch, head) discretization
    # dt_eff = clamp(softplus(dt + dt_bias), floor, max)
    dt_eff = _stable_softplus(dt32 + dt_bias32[None, :])      # [B, num_heads]
    dt_eff = np.clip(dt_eff, time_step_floor, time_step_max)  # [B, num_heads]

    # A is scalar per head: A[h] = -exp(A_log[h])  (A < 0 → stable decay)
    A_per_head = -np.exp(A_log32)                              # [num_heads]

    # Decay = exp(dt_eff * A): scalar per (batch, head)
    decay = np.exp(dt_eff * A_per_head[None, :])               # [B, num_heads]

    # Broadcast per-group B/C to per-head via repeat (g = h // heads_per_group)
    # Vectorised by gather index: group_idx[h] = h // heads_per_group
    group_idx = np.arange(num_heads) // heads_per_group        # [num_heads]
    B_per_head = B32[:, group_idx, :]                          # [B, num_heads, ssm_state]
    C_per_head = C32[:, group_idx, :]                          # [B, num_heads, ssm_state]

    # Input contribution = dt_eff[b, h] * B_per_head[b, h, s] * x[b, h, d]
    # Outer product over (d, s) per (b, h):
    #   shape [B, num_heads, 1, 1] * [B, num_heads, 1, ssm_state] * [B, num_heads, head_dim, 1]
    input_contribution = (
        dt_eff[:, :, None, None]
        * B_per_head[:, :, None, :]
        * x32[:, :, :, None]
    )                                                          # [B, num_heads, head_dim, ssm_state]

    # In-place state update (fp32). Broadcast decay over (head_dim, ssm_state).
    ssm_state[...] = decay[:, :, None, None] * ssm_state + input_contribution

    # Output reduce: y[b, h, d] = sum_s(C[b, h, s] * ssm_state[b, h, d, s]) + D[h] * x[b, h, d]
    y32 = (
        np.einsum("bhs,bhds->bhd", C_per_head, ssm_state)
        + D32[None, :, None] * x32
    )                                                          # [B, num_heads, head_dim]
    return y32


# ── Self-test fixture (deterministic; run with `python3 <this file>`) ──────────
NEMOTRON3_NANO_SHAPES = dict(
    num_heads=64,
    head_dim=64,
    ssm_state=128,
    n_groups=8,
)


def _make_fixture(B: int = 1, *, seed: int = 0):
    """Deterministic-random fixture at Nemotron-3 Nano shapes. Returns a dict
    of all inputs the oracle needs (initial ssm_state = zeros, fp32)."""
    rng = np.random.default_rng(seed)
    H = NEMOTRON3_NANO_SHAPES["num_heads"]
    P = NEMOTRON3_NANO_SHAPES["head_dim"]
    N = NEMOTRON3_NANO_SHAPES["ssm_state"]
    G = NEMOTRON3_NANO_SHAPES["n_groups"]
    return dict(
        x=rng.standard_normal((B, H, P)).astype(np.float32),
        z=rng.standard_normal((B, H, P)).astype(np.float32),
        dt=rng.standard_normal((B, H)).astype(np.float32),
        B_in=rng.standard_normal((B, G, N)).astype(np.float32),
        C_in=rng.standard_normal((B, G, N)).astype(np.float32),
        ssm_state=np.zeros((B, H, P, N), dtype=np.float32),
        # S4D-style A_log init: log(arange(1, H+1)); the brief notes this convention.
        A_log=np.log(np.arange(1, H + 1, dtype=np.float32)),
        dt_bias=rng.standard_normal(H).astype(np.float32) * 0.1,
        D=rng.standard_normal(H).astype(np.float32) * 0.1,
    )


def _main():
    fx = _make_fixture(B=2, seed=42)
    print(f"[oracle] Nemotron-3 Nano shapes: {NEMOTRON3_NANO_SHAPES}")
    print(f"[oracle] inputs:")
    for k, v in fx.items():
        print(f"  {k:10s} {str(v.shape):28s} dtype={v.dtype}  "
              f"min={v.min():+.3f}  max={v.max():+.3f}  mean={v.mean():+.3f}")

    ssm_state_before = fx["ssm_state"].copy()
    y = mamba2_decode_step(**fx)

    print()
    print(f"[oracle] outputs:")
    print(f"  y          {str(y.shape):28s} dtype={y.dtype}  "
          f"min={y.min():+.3f}  max={y.max():+.3f}  mean={y.mean():+.3f}  "
          f"isfinite={np.all(np.isfinite(y))}")
    state_delta = np.abs(fx["ssm_state"] - ssm_state_before).max()
    print(f"  ssm_state' (mutated in place); max|Δ| = {state_delta:+.4f}  "
          f"finite={np.all(np.isfinite(fx['ssm_state']))}")

    # Internal sanity: decay must be in (0, 1] for stable decay.
    A_per_head = -np.exp(fx["A_log"])
    dt_eff = np.clip(
        _stable_softplus(fx["dt"] + fx["dt_bias"][None, :]),
        DEFAULT_TIME_STEP_FLOOR, DEFAULT_TIME_STEP_MAX,
    )
    decay = np.exp(dt_eff * A_per_head[None, :])
    print()
    print(f"[oracle] sanity:")
    print(f"  A_per_head: all < 0 ?            {bool((A_per_head < 0).all())}  "
          f"(min {A_per_head.min():+.4f}, max {A_per_head.max():+.4f})")
    print(f"  dt_eff: all in [floor, max] ?    "
          f"{bool(((dt_eff >= DEFAULT_TIME_STEP_FLOOR) & (dt_eff <= DEFAULT_TIME_STEP_MAX)).all())}  "
          f"(min {dt_eff.min():+.4f}, max {dt_eff.max():+.4f})")
    print(f"  decay: all in (0, 1] ?           "
          f"{bool(((decay > 0) & (decay <= 1)).all())}  "
          f"(min {decay.min():+.4f}, max {decay.max():+.4f})")
    print(f"  state changed from zero ?        {state_delta > 0}")
    print(f"  y is finite ?                    {bool(np.all(np.isfinite(y)))}")

    # Determinism check: re-run with same seed; outputs must bit-equal.
    fx2 = _make_fixture(B=2, seed=42)
    y2 = mamba2_decode_step(**fx2)
    bit_equal = np.array_equal(y, y2)
    print(f"  bit-equal across two runs ?      {bit_equal}")

    print()
    print("[oracle] all internal sanity checks should be True. If any are "
          "False, the oracle has a math bug — fix before using as G0a "
          "ground truth.")


if __name__ == "__main__":
    _main()
