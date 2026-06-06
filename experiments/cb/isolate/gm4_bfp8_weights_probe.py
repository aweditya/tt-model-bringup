"""Isolation probe: bfloat16 vs bfloat8_b WEIGHT dtype on representative
Gemma 4 decoder matmuls.

Round 8 lever (profile-driven). The fresh DRAM-BW breakdown (via
`experiments/utils/dram_bw_matmul_breakdown.py` on the Round 7 Tracy CSV)
showed that **Matmul dominates the per-forward PM-BANDWIDTH budget at
99.5% (43.487 ms of 43.695 ms)**, with the MLP `[32, 3840] x [3840, 3840]`
triplet alone accounting for **71% of all matmul PM-BW** (144 calls/forward
across gate_proj + up_proj + down_proj × 48 layers).

This confirms Round 7's BW-bound diagnosis quantitatively. The cheapest BW-
reduction lever that DOES NOT require Megatron-TP architectural surgery
(distributed RMSNorm + reduce_scatter + global_cb prefetcher = multi-week)
is `bfloat8_b` weights: same op-count, same program-config, same kernel —
just half the bytes read from DRAM per matmul. Precision is preserved in
the activation pipeline (still bf16) and the accumulator (`fp32_dest_acc_en
=True`).

Precedent in this repo: `experiments/serve/server_35b_ttnn.py:320-336`
ships `bfloat8_b` for the heavy MoE expert weights with a documented
PCC=0.999903 vs bf16 reference; production code path. Llama 70B Galaxy
ships `bfloat8_b` for MLP weights end-to-end (model_config: BFP8_MM_OUTPUT).

Gate: cos(bf16, bfp8) >= 0.999 on production-scale matmul shapes, max|delta|
< 0.5 (forgiving for [3840] reduction at this scale). If all 5 shapes PASS,
we ship the lever in `server_gemma4_unified_ttnn.py` MLP weights only first
(largest BW share, smallest blast radius); next round expands to Q/K/V/O if
needed.

Forks: `experiments/cb/isolate/gm4_hifi2_matmul_probe.py` (probe scaffold).
"""
from __future__ import annotations

import sys
import time

import torch
import ttnn

HIDDEN = 3840


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _cos(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def _cfg():
    # Match production HIFI4 in server_gemma4_unified_ttnn.py: HiFi4 +
    # fp32_dest_acc_en=True. bfp8 weight is read once at matmul invocation;
    # accumulator precision is untouched.
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def _test_shape(name, M, K, N, x_tt_factory, w_tt_factory_bf16, w_tt_factory_bfp8,
                to_torch, expected_shape, x_scale, w_scale):
    log(f"--- {name}: [{M},{K}] x [{K},{N}] = [{M},{N}] ---")
    torch.manual_seed(0)
    x_t = torch.randn(M, K, dtype=torch.float32) * x_scale
    w_t = torch.randn(K, N, dtype=torch.float32) * w_scale
    y_ref = (x_t @ w_t)  # fp32 ground truth

    x_tt = x_tt_factory(x_t)
    w_tt_bf16 = w_tt_factory_bf16(w_t)
    w_tt_bfp8 = w_tt_factory_bfp8(w_t)

    y_bf16_tt = ttnn.matmul(x_tt, w_tt_bf16, compute_kernel_config=_cfg())
    y_bf16 = to_torch(y_bf16_tt, expected_shape)
    ttnn.deallocate(y_bf16_tt)

    y_bfp8_tt = ttnn.matmul(x_tt, w_tt_bfp8, compute_kernel_config=_cfg())
    y_bfp8 = to_torch(y_bfp8_tt, expected_shape)
    ttnn.deallocate(y_bfp8_tt)

    ttnn.deallocate(x_tt)
    ttnn.deallocate(w_tt_bf16)
    ttnn.deallocate(w_tt_bfp8)

    cos_bf16_ref = _cos(y_bf16, y_ref)
    cos_bfp8_ref = _cos(y_bfp8, y_ref)
    cos_pair = _cos(y_bf16, y_bfp8)
    max_abs = float((y_bf16 - y_bfp8).abs().max())
    mad = float((y_bf16 - y_bfp8).abs().mean())
    # Magnitude ratio to detect scale shifts (the bf8 shared-exponent failure mode)
    mag_bf16 = float(y_bf16.float().abs().mean())
    mag_bfp8 = float(y_bfp8.float().abs().mean())
    ratio = (mag_bfp8 / mag_bf16) if mag_bf16 > 0 else float("inf")

    log(f"  cos(bf16, fp32_ref)  = {cos_bf16_ref:.7f}")
    log(f"  cos(bfp8, fp32_ref)  = {cos_bfp8_ref:.7f}  (delta vs bf16: {cos_bfp8_ref - cos_bf16_ref:+.7f})")
    log(f"  cos(bf16, bfp8)      = {cos_pair:.7f}")
    log(f"  max|bf16 - bfp8|     = {max_abs:.6f}")
    log(f"  mean|bf16 - bfp8|    = {mad:.6f}")
    log(f"  magnitude ratio     = {ratio:.4f}  (1.0 = no scale shift)")

    ok = cos_pair >= 0.999 and max_abs < 0.5 and 0.95 < ratio < 1.05
    log(f"  PASS: {'yes' if ok else 'NO'}")
    return ok


def main(state=None):
    if state is None:
        log("ERR: probe requires a harness mesh; run via gm4 dev harness.")
        return 1
    device = state.mesh
    mapper = ttnn.ReplicateTensorToMesh(device)
    composer = ttnn.ConcatMeshToTensor(device, dim=0)

    def to_tt_bf16(t):
        return ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=device, mesh_mapper=mapper,
        )

    def to_tt_bfp8(t):
        return ttnn.from_torch(
            t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
            device=device, mesh_mapper=mapper,
        )

    def to_torch(tt, expected_shape):
        arr = ttnn.to_torch(tt, mesh_composer=composer)
        return arr[: expected_shape[0]].reshape(*expected_shape)

    results = {}

    # MLP gate / up / down (same per-chip shape after TP=4 sharding).
    results["gate_proj"] = _test_shape(
        "gate_proj", 1, 3840, 3840, to_tt_bf16, to_tt_bf16, to_tt_bfp8,
        to_torch, (1, 3840), x_scale=0.5, w_scale=0.05,
    )
    results["up_proj"] = _test_shape(
        "up_proj", 1, 3840, 3840, to_tt_bf16, to_tt_bf16, to_tt_bfp8,
        to_torch, (1, 3840), x_scale=0.5, w_scale=0.05,
    )
    results["down_proj"] = _test_shape(
        "down_proj", 1, 3840, 3840, to_tt_bf16, to_tt_bf16, to_tt_bfp8,
        to_torch, (1, 3840), x_scale=0.5, w_scale=0.05,
    )

    # Attention projections (smaller per-chip shape; lower DRAM share but
    # still worth checking precision-wise in case we expand later).
    results["q_proj_sliding"] = _test_shape(
        "q_proj_sliding", 1, 3840, 1024, to_tt_bf16, to_tt_bf16, to_tt_bfp8,
        to_torch, (1, 1024), x_scale=0.5, w_scale=0.05,
    )
    results["o_proj_sliding"] = _test_shape(
        "o_proj_sliding", 1, 1024, 3840, to_tt_bf16, to_tt_bf16, to_tt_bfp8,
        to_torch, (1, 3840), x_scale=0.5, w_scale=0.05,
    )

    log("")
    log("=" * 70)
    all_pass = all(results.values())
    for k, v in results.items():
        log(f"  {k:24s}: {'PASS' if v else 'FAIL'}")
    log(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    log("=" * 70)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
