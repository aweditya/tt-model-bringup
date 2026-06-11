#!/usr/bin/env python3
"""MM7 v0.1.2.c debug — numpy-only reference for the L0 Mamba2 forward.

When the on-device smoke fails, this script verifies whether the chain
math (SSD → MambaRMSNormGated) matches HF on the host. Lets us localize:
- If numpy_norm == HF_norm: the bug is in our TT implementation
- If numpy_norm != HF_norm: our numpy oracle has a different convention
  than the HF stub/oracle

Tests BOTH norm-before-gate orderings to see which matches the oracle.
"""
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "utils"))
from mamba2_numpy_oracle import mamba2_decode_step  # noqa

ORACLE_DIR = PROJECT_ROOT / ".cache" / "hf_oracle_nemotron3_nano"
EPS = 1e-5
D_INNER = 4096
CONV_DIM = 6144
NUM_HEADS = 64
HD = 64
NG = 8
SS = 128
S = 5
L0 = 0
GROUP_SIZE = D_INNER // NG  # 512


def cos(a, b):
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def silu(x):
    return x / (1.0 + np.exp(-x))


def rmsnorm_gated_stub(x, w, z, eps, group_size):
    """Matches our mamba_ssm CPU stub: group RMSNorm + weight, then gate."""
    new_shape = x.shape[:-1] + (x.shape[-1] // group_size, group_size)
    xg = x.reshape(new_shape)
    var = (xg ** 2).mean(axis=-1, keepdims=True)
    xg_norm = xg * (1.0 / np.sqrt(var + eps))
    out = xg_norm.reshape(x.shape) * w
    out = out * silu(z)
    return out


def rmsnorm_gated_gate_first(x, w, z, eps, group_size):
    """Real mamba_ssm semantics when norm_before_gate=False:
       gate first, then group RMSNorm, then weight."""
    x_gated = x * silu(z)
    new_shape = x_gated.shape[:-1] + (x_gated.shape[-1] // group_size, group_size)
    xg = x_gated.reshape(new_shape)
    var = (xg ** 2).mean(axis=-1, keepdims=True)
    xg_norm = xg * (1.0 / np.sqrt(var + eps))
    return xg_norm.reshape(x_gated.shape) * w


def main() -> int:
    # ── Load HF oracle ─────────────────────────────────────────
    in_proj_hf = np.load(ORACLE_DIR / "L0_in_proj.npy")[0]
    conv1d_hf = np.load(ORACLE_DIR / "L0_conv1d.npy")[0]
    norm_hf = np.load(ORACLE_DIR / "L0_norm.npy")[0]

    # ── Load weights ───────────────────────────────────────────
    snap = next(Path.home().joinpath(
        ".cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots"
    ).glob("*"))
    weights = {}
    for k in ["dt_bias", "A_log", "D", "norm.weight"]:
        for s in sorted(snap.glob("*.safetensors")):
            with safe_open(s, framework="pt") as f:
                key = f"backbone.layers.{L0}.mixer.{k}"
                if key in f.keys():
                    weights[k] = f.get_tensor(key).float().numpy()
                    break

    # ── Reconstruct the path: split in_proj, post-conv1d-silu split ──
    gate = in_proj_hf[:, :D_INNER]
    dt = in_proj_hf[:, D_INNER + CONV_DIM:D_INNER + CONV_DIM + NUM_HEADS]
    # HF L0_conv1d shape is [conv_dim, S+pad]; transpose + causal slice
    conv_out_causal = conv1d_hf.T[:S, :]
    xBC_silu = silu(conv_out_causal)
    x_inner = xBC_silu[:, :D_INNER].reshape(S, NUM_HEADS, HD)
    B_inner = xBC_silu[:, D_INNER:D_INNER + NG * SS].reshape(S, NG, SS)
    C_inner = xBC_silu[:, D_INNER + NG * SS:].reshape(S, NG, SS)
    z_per_pos = gate.reshape(S, NUM_HEADS, HD)

    # ── SSD loop ───────────────────────────────────────────────
    ssm_state = np.zeros((1, NUM_HEADS, HD, SS), dtype=np.float32)
    y_per_pos = []
    for p in range(S):
        fx = dict(
            x=x_inner[p:p + 1],
            z=z_per_pos[p:p + 1],
            dt=dt[p:p + 1],
            dt_bias=weights["dt_bias"],
            A_log=weights["A_log"],
            D=weights["D"],
            B_in=B_inner[p:p + 1],
            C_in=C_inner[p:p + 1],
            ssm_state=ssm_state.copy(),
        )
        y = mamba2_decode_step(**fx)
        ssm_state = fx["ssm_state"]
        y_per_pos.append(y[0])

    y_post_ssd_np = np.stack(y_per_pos, axis=0)
    y_flat_np = y_post_ssd_np.reshape(S, NUM_HEADS * HD)
    print(f"y_post_ssd numpy shape: {y_flat_np.shape}  "
          f"range [{y_flat_np.min():.3f}, {y_flat_np.max():.3f}]")

    # ── Compare against HF y_pre_norm (now captured by the oracle) ──
    y_pre_norm_hf_path = ORACLE_DIR / "L0_y_pre_norm.npy"
    if y_pre_norm_hf_path.exists():
        y_hf = np.load(y_pre_norm_hf_path)
        if y_hf.ndim == 3 and y_hf.shape[0] == 1:
            y_hf = y_hf[0]
        print(f"\ny_hf shape: {y_hf.shape}  "
              f"range [{y_hf.min():.3f}, {y_hf.max():.3f}]")
        cos_y = cos(y_flat_np, y_hf)
        print(f"cos(numpy oracle y, HF y_pre_norm) = {cos_y:.6f}")
        # Per-position cosine to see if drift is monotonic
        for p in range(S):
            cos_p = cos(y_flat_np[p], y_hf[p])
            print(f"  pos {p}: cos = {cos_p:.6f}")
    else:
        print(f"\n(L0_y_pre_norm.npy not present — re-run oracle with the hook)")

    # ── Compare both norm orderings vs HF ─────────────────────
    norm_w = weights["norm.weight"]
    norm_stub = rmsnorm_gated_stub(y_flat_np, norm_w, gate, EPS, GROUP_SIZE)
    norm_gf = rmsnorm_gated_gate_first(y_flat_np, norm_w, gate, EPS, GROUP_SIZE)

    cos_stub = cos(norm_stub, norm_hf)
    cos_gf = cos(norm_gf, norm_hf)
    print(f"\ncos(norm-first-then-gate [stub style], HF L0_norm) = {cos_stub:.6f}")
    print(f"cos(gate-first-then-norm [real mamba_ssm],   HF L0_norm) = {cos_gf:.6f}")

    print(f"\nnorm_hf  range [{norm_hf.min():.3f}, {norm_hf.max():.3f}]")
    print(f"norm_stub range [{norm_stub.min():.3f}, {norm_stub.max():.3f}]")
    print(f"norm_gf   range [{norm_gf.min():.3f}, {norm_gf.max():.3f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
