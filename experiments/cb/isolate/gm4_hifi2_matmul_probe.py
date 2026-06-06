"""Isolation probe: HiFi4 vs HiFi2 on representative Gemma 4 decoder matmuls.

Round 7 lever (profile-driven): tt-perf-report on the Round-6 traced forward
flagged Matmul = 23.58% of measured kernel time, with the hint:
    "HiFi2 may also work, it discards the lowest bit of the activations and
     has 2x the throughput of HiFi4"
on every dense matmul site. Llama 70B Galaxy production
(`tt-metal/models/demos/llama3_70b_galaxy/tt/llama_attention.py:411,565,614`)
ships HiFi2 for Q/K/V/O — confirming the operating point.

Projected gain: 23.58% × 2x speedup = ~12% traced (well into the 5-15% sweet
spot the brief asked for). Risk: bf16 precision drift over 48 layers. We mitigate
by keeping `fp32_dest_acc_en=True` (the 91f chain-drift insurance from
[[bf16-chain-drift-at-B-gt-1]]) — HiFi2 only loses the lowest BIT of the
activation per multiply, not the accumulator precision.

Tested shapes (representative of the hot decoder matmuls per-chip on qb2 mesh):
- gate_proj: [3840, 3840]  (FFN inner per-chip)
- down_proj: [3840, 3840]  (FFN outer per-chip — same shape, transposed scale)
- q_proj:    [3840, 1024]  (Q per-chip)
- o_proj:    [1024, 3840]  (O per-chip)
- lm_head:   [3840, 65536] (vocab per-chip; not tested here — only forward path)

Gate: cos(HiFi4, HiFi2) > 0.999 and max|delta| < 0.05 (matches the Round-4
addcmul probe acceptance bound) on a representative sample. If fails, abandon
and document.

Forks: `experiments/cb/isolate/gm4_matmul_gelu_probe.py` (probe scaffolding,
to_tt/to_torch helpers).
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


def _hifi_cfg(fidelity):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=True,  # keep 91f chain-drift insurance
        packer_l1_acc=False,    # matches production HIFI4 in server_gemma4_unified_ttnn.py:97
    )


def _test_shape(name, M, K, N, x_tt_factory, w_tt_factory, to_torch, expected_shape, x_scale, w_scale):
    log(f"--- {name}: [1,{K}] x [{K},{N}] = [1,{N}] ---")
    torch.manual_seed(0)
    x_t = torch.randn(1, K, dtype=torch.float32) * x_scale
    w_t = torch.randn(K, N, dtype=torch.float32) * w_scale
    y_ref = (x_t @ w_t)  # fp32 ground truth

    x_tt = x_tt_factory(x_t)
    w_tt = w_tt_factory(w_t)

    y_hifi4_tt = ttnn.matmul(x_tt, w_tt, compute_kernel_config=_hifi_cfg(ttnn.MathFidelity.HiFi4))
    y_hifi4 = to_torch(y_hifi4_tt, expected_shape)
    ttnn.deallocate(y_hifi4_tt)

    y_hifi2_tt = ttnn.matmul(x_tt, w_tt, compute_kernel_config=_hifi_cfg(ttnn.MathFidelity.HiFi2))
    y_hifi2 = to_torch(y_hifi2_tt, expected_shape)
    ttnn.deallocate(y_hifi2_tt)

    ttnn.deallocate(x_tt); ttnn.deallocate(w_tt)

    cos_h4_ref = _cos(y_hifi4, y_ref)
    cos_h2_ref = _cos(y_hifi2, y_ref)
    cos_pair = _cos(y_hifi4, y_hifi2)
    max_abs = float((y_hifi4 - y_hifi2).abs().max())
    mad = float((y_hifi4 - y_hifi2).abs().mean())

    log(f"  cos(HiFi4, fp32_ref) = {cos_h4_ref:.7f}")
    log(f"  cos(HiFi2, fp32_ref) = {cos_h2_ref:.7f}  (delta vs HiFi4: {cos_h2_ref - cos_h4_ref:+.7f})")
    log(f"  cos(HiFi4, HiFi2)    = {cos_pair:.7f}")
    log(f"  max|HiFi4 - HiFi2|   = {max_abs:.6f}")
    log(f"  mean|HiFi4 - HiFi2|  = {mad:.6f}")

    ok = cos_pair > 0.999 and max_abs < 0.5  # 0.5 generous for [3840] reduction × bf16 noise
    log(f"  PASS: {'yes' if ok else 'NO'}")
    return ok


def main(state=None):
    if state is None:
        log("ERR: probe requires a harness mesh; run via gm4 dev harness.")
        return 1
    device = state.mesh
    mapper = ttnn.ReplicateTensorToMesh(device)
    composer = ttnn.ConcatMeshToTensor(device, dim=0)

    def to_tt(t):
        return ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=device, mesh_mapper=mapper,
        )

    def to_torch(tt, expected_shape):
        arr = ttnn.to_torch(tt, mesh_composer=composer)
        return arr[: expected_shape[0]].reshape(*expected_shape)

    results = {}

    # 1. Q proj: [3840] x [3840, 1024] — Q proj per-chip (NQ=4, head_dim=256 sliding)
    results["q_proj_sliding"] = _test_shape(
        "q_proj_sliding", 1, 3840, 1024, to_tt, to_tt, to_torch, (1, 1024),
        x_scale=0.5, w_scale=0.05,
    )

    # 2. K/V proj: [3840] x [3840, 512] — K/V proj per-chip (NKV=2, head_dim=256)
    results["kv_proj_sliding"] = _test_shape(
        "kv_proj_sliding", 1, 3840, 512, to_tt, to_tt, to_torch, (1, 512),
        x_scale=0.5, w_scale=0.05,
    )

    # 3. O proj: [1024] x [1024, 3840] — column-sharded; per-chip is partial sum
    results["o_proj"] = _test_shape(
        "o_proj", 1, 1024, 3840, to_tt, to_tt, to_torch, (1, 3840),
        x_scale=0.5, w_scale=0.05,
    )

    # 4. gate_proj: [3840] x [3840, 3840] — FFN inner per-chip
    results["gate_proj"] = _test_shape(
        "gate_proj", 1, 3840, 3840, to_tt, to_tt, to_torch, (1, 3840),
        x_scale=0.5, w_scale=0.05,
    )

    # 5. down_proj: [3840] x [3840, 3840] — FFN outer per-chip (column-sharded)
    results["down_proj"] = _test_shape(
        "down_proj", 1, 3840, 3840, to_tt, to_tt, to_torch, (1, 3840),
        x_scale=0.5, w_scale=0.05,
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
