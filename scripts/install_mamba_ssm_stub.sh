#!/usr/bin/env bash
# Install a pure-PyTorch stub of `mamba_ssm` into the project venv so the
# Nemotron-3 Nano HF modeling code can import without the CUDA-only
# mamba-ssm package. We only need to satisfy ONE symbol: `rmsnorm_fn`
# from `mamba_ssm.ops.triton.layernorm_gated`. The other Mamba2 entry
# points (`selective_state_update`, `mamba_chunk_scan_combined`,
# `mamba_split_conv1d_scan_combined`) get set to None when
# `is_mamba_2_ssm_available()` returns False — they're not on the
# code path we exercise.
#
# Usage (from project root, on the QuietBox):
#   bash scripts/install_mamba_ssm_stub.sh
#
# Reverse with: rm -rf .venv/lib/python3.10/site-packages/mamba_ssm

set -e

VENV_ROOT="${VENV_ROOT:-.venv}"
PY_VERSION="${PY_VERSION:-python3.10}"
SITE_PACKAGES="$VENV_ROOT/lib/$PY_VERSION/site-packages"

if [[ ! -d "$SITE_PACKAGES" ]]; then
    echo "error: $SITE_PACKAGES not found — run from project root with the venv activated"
    exit 1
fi

STUB_DIR="$SITE_PACKAGES/mamba_ssm"
mkdir -p "$STUB_DIR/ops/triton"

cat > "$STUB_DIR/__init__.py" <<'PY'
"""Stub `mamba_ssm` for CPU-only Nemotron-3 Nano HF oracle generation.

Only provides the ONE function the modeling code hard-imports:
`mamba_ssm.ops.triton.layernorm_gated.rmsnorm_fn`. The other Mamba2
fast-path entry points are guarded by `is_mamba_2_ssm_available()`
checks and get set to None when this stub package's modules don't
declare them — exactly the path we want on CPU.
"""
PY

cat > "$STUB_DIR/ops/__init__.py" <<'PY'
PY

cat > "$STUB_DIR/ops/triton/__init__.py" <<'PY'
PY

cat > "$STUB_DIR/ops/triton/layernorm_gated.py" <<'PY'
"""Pure-PyTorch RMSNorm stub used by Nemotron-H modeling code (CPU path).

`rmsnorm_fn` mirrors the signature mamba-ssm's Triton kernel exposes.
We implement the math in standard PyTorch — slower than the kernel
but correct, and only used for one-shot oracle activation generation
on CPU (host).
"""
import torch


def rmsnorm_fn(
    x,
    weight,
    bias=None,
    residual=None,
    eps: float = 1e-6,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    is_rms_norm: bool = True,
    group_size=None,
    norm_before_gate: bool = True,
    z=None,
):
    """RMSNorm (and gated variant when z is given) — pure PyTorch.

    Matches the Triton kernel's behavior closely enough for cosine-correctness
    sanity checks during Nemotron-3 Nano bringup. Not exhaustively tested
    against the kernel; if a downstream cosine ladder gates fail strangely,
    this is the first place to look.
    """
    orig_dtype = x.dtype
    if residual is not None:
        x = x + (residual.to(torch.float32) if residual_in_fp32 else residual)
        residual_out = x
    else:
        residual_out = x

    if group_size is None or group_size <= 0:
        # Standard RMSNorm over the last dim.
        x_fp32 = x.to(torch.float32)
        var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_fp32 * torch.rsqrt(var + eps)
        out = x_norm.to(orig_dtype)
        if weight is not None:
            out = out * weight
        if bias is not None:
            out = out + bias
    else:
        # Group RMSNorm: split the last dim into G groups, RMSNorm each
        # group independently, scale/shift per group.
        last = x.shape[-1]
        if last % group_size != 0:
            raise ValueError(
                f"last dim {last} not divisible by group_size {group_size}")
        x_fp32 = x.to(torch.float32)
        new_shape = x.shape[:-1] + (last // group_size, group_size)
        xg = x_fp32.view(*new_shape)
        var = xg.pow(2).mean(dim=-1, keepdim=True)
        xg_norm = xg * torch.rsqrt(var + eps)
        out = xg_norm.view(*x.shape).to(orig_dtype)
        if weight is not None:
            out = out * weight
        if bias is not None:
            out = out + bias

    # Gated RMSNorm: y = norm(x) * silu(z).
    if z is not None:
        out = out * torch.nn.functional.silu(z)

    if prenorm:
        return out, residual_out
    return out
PY

echo "mamba_ssm stub installed at: $STUB_DIR"
echo "verify with:"
echo "  $VENV_ROOT/bin/python -c 'from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn; print(rmsnorm_fn)'"
