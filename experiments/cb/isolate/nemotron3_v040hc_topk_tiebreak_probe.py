#!/usr/bin/env python3
"""MM7 v0.4.0h.c — ttnn.topk tie-break tester for the MoE router.

Forks v040hb. Tests four candidates for the EXACT same problem the
production v040hb router solves (sigmoid + bias + topk-6 over 128 experts),
ranking them by: (a) per-row idx-set match vs numpy.argpartition,
(b) weights cos.

Variants:
  A. baseline   : current production. ttnn.topk(bf16 scores_biased)
  B. idx_offset : (scores_biased - idx*EPS) so ties break to lowest idx.
                  Per-expert offset table replicated and ttnn.subtract'd
                  before the topk. Matches numpy argpartition tie-break.
  C. fp32_promote: ttnn.typecast scores_biased → fp32 (DataType.float32)
                  before topk. Tests whether fp32 compare avoids the
                  sign-magnitude SFPSWAP asymmetry described in #20625.
  D. sorted     : ttnn.topk(..., sorted=True) — sanity that the flag is
                  already on in our prod call.

Rationale (cited):
  - tt-metal #20625 (rdjogoTT): the HW SFPSWAP compares by magnitude
    then sign separately; this is asymmetric for positive/negative
    values and NOT consistent with stable sorting.
  - tt-metal #33492: even after PR #31989 added a stable LLK flag,
    ttnn.sort's stable=True still returns wrong indices.
  - ttnn.topk currently has NO stable flag exposed.
  - DeepSeek-V3 demo (tt-metal models/demos/deepseek_v3/tests/test_moe.py
    line 209-211, line 274) runs all MoE forward-pass tests with
    `topk_fallback=True` — torch.topk via host bitonic — not ttnn.topk.

Run:
  ssh qb1 'touch ~/tt-xla/.cache/nm3_runtime/trig/v040hc_topk_tiebreak_probe'
  ssh qb1 'cat ~/tt-xla/.cache/nm3_runtime/trig/last.log'
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

B = 1
S_PADDED = 8
NUM_EXPERTS = 128
TOP_K = 6
ROUTED_SCALING = 2.5


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def numpy_reference(scores_np, bias_np):
    scores_biased = scores_np + bias_np[None, None, :]
    topk_idx = np.argpartition(
        -scores_biased.reshape(-1, NUM_EXPERTS), TOP_K, axis=-1
    )[:, :TOP_K].reshape(B, S_PADDED, TOP_K)
    rows = np.arange(B * S_PADDED).reshape(B, S_PADDED, 1)
    topk_weights = np.take_along_axis(
        scores_np.reshape(B * S_PADDED, NUM_EXPERTS),
        topk_idx.reshape(B * S_PADDED, TOP_K),
        axis=-1,
    ).reshape(B, S_PADDED, TOP_K)
    denom = topk_weights.sum(axis=-1, keepdims=True) + 1e-20
    topk_weights = topk_weights / denom * ROUTED_SCALING
    return topk_idx, topk_weights


def _to_tt_replicated(arr, ttnn, mesh, dtype, layout=None):
    layout = layout or ttnn.TILE_LAYOUT
    return ttnn.from_torch(
        torch.from_numpy(np.ascontiguousarray(arr.astype(np.float32))),
        dtype=dtype, layout=layout, device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _to_np(t, ttnn, mesh):
    arr = ttnn.to_torch(
        t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
    )
    return arr[:1].float().numpy()


def device_topk_baseline(scores_biased_tt, ttnn):
    """A. baseline — exactly what production does."""
    return ttnn.topk(scores_biased_tt, k=TOP_K, dim=-1)


def device_topk_idx_offset(scores_biased_tt, idx_offset_tt, ttnn):
    """B. idx_offset — subtract idx*eps so smaller-idx wins ties."""
    offset_scores_tt = ttnn.subtract(scores_biased_tt, idx_offset_tt)
    out = ttnn.topk(offset_scores_tt, k=TOP_K, dim=-1)
    ttnn.deallocate(offset_scores_tt)
    return out


def device_topk_fp32(scores_biased_tt, ttnn):
    """C. fp32_promote — cast to fp32 before topk."""
    scores_fp32 = ttnn.typecast(scores_biased_tt, ttnn.float32)
    out = ttnn.topk(scores_fp32, k=TOP_K, dim=-1)
    ttnn.deallocate(scores_fp32)
    return out


def device_topk_sorted(scores_biased_tt, ttnn):
    """D. sorted=True (production already passes default; explicit here)."""
    return ttnn.topk(scores_biased_tt, k=TOP_K, dim=-1, sorted=True)


def evaluate(variant_name, ref_idx, dev_idx, ref_weights, dev_weights):
    """Return (idx_match_count, idx_total, weights_cos, weights_mad)."""
    idx_match = 0
    idx_total = B * S_PADDED
    for b in range(B):
        for s in range(S_PADDED):
            ref_set = set(ref_idx[b, s].tolist())
            dev_set = set(dev_idx[b, s].tolist())
            if ref_set == dev_set:
                idx_match += 1
    # weights cos after sorting by index ascending
    ref_sorted = np.zeros_like(ref_weights)
    dev_sorted = np.zeros_like(dev_weights)
    for b in range(B):
        for s in range(S_PADDED):
            ref_order = np.argsort(ref_idx[b, s])
            dev_order = np.argsort(dev_idx[b, s])
            ref_sorted[b, s] = ref_weights[b, s, ref_order]
            dev_sorted[b, s] = dev_weights[b, s, dev_order]
    cos = float(
        np.dot(ref_sorted.reshape(-1), dev_sorted.reshape(-1))
        / (np.linalg.norm(ref_sorted) * np.linalg.norm(dev_sorted) + 1e-12)
    )
    mad = float(np.mean(np.abs(ref_sorted - dev_sorted)))
    log(f"  {variant_name:14s}: idx_set_match={idx_match}/{idx_total}  cos={cos:.6f}  mad={mad:.4e}")
    return idx_match, idx_total, cos, mad


def main(state=None) -> int:
    import ttnn

    own_mesh = state is None
    if state is not None and getattr(state, "mesh", None) is not None:
        mesh = state.mesh
        log("[harness] reusing live mesh ✓")
    else:
        log("opening (1,4) mesh…")
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(
            ttnn.MeshShape(1, 4),
            l1_small_size=65536,
            trace_region_size=50_000_000,
        )

    try:
        # Distribution chosen to match production:
        # scores = sigmoid(matmul) ∈ (0, 1); bias is small zero-mean.
        rng = np.random.default_rng(seed=99)
        # Simulate sigmoid output — bf16-quantize after creation by
        # uploading then reading back. For the reference, use the
        # ORIGINAL fp32 values so we measure "did ttnn match true argpartition".
        scores_np = rng.uniform(0.0, 1.0, size=(B, S_PADDED, NUM_EXPERTS)).astype(np.float32)
        bias_np = (rng.standard_normal((NUM_EXPERTS,), dtype=np.float32) * 0.1)

        ref_idx, ref_weights = numpy_reference(scores_np, bias_np)
        log(f"numpy ref idx shape: {ref_idx.shape}")

        # Also build a "bf16-aware" reference: round scores+bias to bf16
        # FIRST, then argpartition. This is the ACHIEVABLE ground truth
        # given the dtype contract of ttnn.topk.
        def to_bf16(x: np.ndarray) -> np.ndarray:
            return torch.from_numpy(np.ascontiguousarray(x.astype(np.float32))).to(torch.bfloat16).to(torch.float32).numpy()
        scores_bf16 = to_bf16(scores_np)
        bias_bf16 = to_bf16(bias_np)
        scores_biased_bf16 = to_bf16(scores_bf16 + bias_bf16[None, None, :])
        ref_bf16_idx = np.argpartition(
            -scores_biased_bf16.reshape(-1, NUM_EXPERTS), TOP_K, axis=-1
        )[:, :TOP_K].reshape(B, S_PADDED, TOP_K)
        # Count exact ties within top-K boundary of the bf16-rounded
        # biased scores (the rows where SOME tie-break decision is being
        # made at all).
        tied_rows = 0
        for b_i in range(B):
            for s_i in range(S_PADDED):
                row = scores_biased_bf16[b_i, s_i]
                sorted_vals = np.sort(row)[::-1]
                kth = sorted_vals[TOP_K - 1]
                ties_at_kth = int(np.sum(row == kth))
                if ties_at_kth > 1:
                    tied_rows += 1
        log(f"bf16-aware: rows with EXACT tie at K-th rank: {tied_rows}/{B*S_PADDED}")

        # Diagnostic: HOST path (production default) — bf16 scores
        # readback from device + FP32 bias added on host. Mimics line
        # 2761-2767 of server_nemotron3_nano_ttnn.py.
        host_scores_for_choice = scores_bf16 + bias_np[None, None, :]
        host_idx = np.argpartition(
            -host_scores_for_choice.reshape(-1, NUM_EXPERTS),
            TOP_K, axis=-1,
        )[:, :TOP_K].reshape(B, S_PADDED, TOP_K)
        host_match = sum(
            1 for b_i in range(B) for s_i in range(S_PADDED)
            if set(host_idx[b_i, s_i].tolist()) == set(ref_idx[b_i, s_i].tolist())
        )
        log(f"HOST path (bf16 scores + fp32 bias) vs full-fp32 ref: {host_match}/{B*S_PADDED}")
        # Diagnostic: HOST path vs bf16-aware ref (what device sees).
        host_vs_bf16ref = sum(
            1 for b_i in range(B) for s_i in range(S_PADDED)
            if set(host_idx[b_i, s_i].tolist()) == set(ref_bf16_idx[b_i, s_i].tolist())
        )
        log(f"HOST path vs bf16-aware ref: {host_vs_bf16ref}/{B*S_PADDED}")

        scores_tt = _to_tt_replicated(scores_np, ttnn, mesh, ttnn.bfloat16)
        bias_tt = _to_tt_replicated(
            bias_np.reshape(1, 1, NUM_EXPERTS), ttnn, mesh, ttnn.bfloat16,
        )
        scores_biased_tt = ttnn.add(scores_tt, bias_tt)

        # idx_offset table: [1,1,128] = [0, eps, 2eps, …]. Subtracted from
        # scores_biased so position i in case of EXACT tie gets penalised
        # by i*EPS, breaking ties to the lowest index — matching numpy
        # argpartition's behaviour. EPS chosen << smallest meaningful
        # score gap but > bf16 ULP at score magnitudes ~1.
        EPS_VARIANTS = [1.0e-4, 1.0e-3, 1.0e-2]
        OFFSET_EPS = 1.0e-3  # default — tested at three magnitudes below
        idx_offset_np = (np.arange(NUM_EXPERTS, dtype=np.float32) * OFFSET_EPS).reshape(1, 1, NUM_EXPERTS)
        idx_offset_tt = _to_tt_replicated(idx_offset_np, ttnn, mesh, ttnn.bfloat16)

        log("")
        log("=" * 70)
        log("VARIANT EVALUATION (B*S = 8 rows, K=6, 128 experts)")
        log("=" * 70)

        def run_and_eval(name, fn, *args):
            vals_tt, idxs_tt = fn(scores_biased_tt, *args, ttnn)
            idxs_np = _to_np(idxs_tt, ttnn, mesh).astype(np.int64)
            # Weights: numpy reference's weights are scores[topk_idxs]
            # gathered + normalised. For dev, gather UNbiased scores at
            # the device-picked idxs (host-side, since we only care about
            # IDX correctness here).
            dev_weights = np.take_along_axis(
                scores_np.reshape(B * S_PADDED, NUM_EXPERTS),
                idxs_np.reshape(B * S_PADDED, TOP_K),
                axis=-1,
            ).reshape(B, S_PADDED, TOP_K)
            denom = dev_weights.sum(axis=-1, keepdims=True) + 1e-20
            dev_weights = dev_weights / denom * ROUTED_SCALING
            r = evaluate(name, ref_idx, idxs_np, ref_weights, dev_weights)
            ttnn.deallocate(vals_tt)
            ttnn.deallocate(idxs_tt)
            return r

        # Variant E: re-add bias with HIFI4 + fp32_dest_acc_en compute_kernel_config.
        # Tests whether the bf16-tie comes from the ADD precision (not topk).
        try:
            HIFI4 = ttnn.init_device_compute_kernel_config(
                mesh.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
            scores_biased_hifi4_tt = ttnn.add(
                scores_tt, bias_tt, compute_kernel_config=HIFI4,
            )
            vals_tt, idxs_tt = ttnn.topk(scores_biased_hifi4_tt, k=TOP_K, dim=-1)
            ttnn.deallocate(scores_biased_hifi4_tt)
            idxs_np = _to_np(idxs_tt, ttnn, mesh).astype(np.int64)
            dev_weights = np.take_along_axis(
                scores_np.reshape(B * S_PADDED, NUM_EXPERTS),
                idxs_np.reshape(B * S_PADDED, TOP_K),
                axis=-1,
            ).reshape(B, S_PADDED, TOP_K)
            denom = dev_weights.sum(axis=-1, keepdims=True) + 1e-20
            dev_weights = dev_weights / denom * ROUTED_SCALING
            e = evaluate("E add_HIFI4", ref_idx, idxs_np, ref_weights, dev_weights)
            ttnn.deallocate(vals_tt); ttnn.deallocate(idxs_tt)
        except Exception as ex:
            log(f"  E add_HIFI4    : SKIPPED ({type(ex).__name__}: {ex})")
            e = (0, 8, 0.0, 0.0)

        # Variant F: upload bias as bf16-rounded value of fp32 bias
        # (control: what device sees TODAY).
        a = run_and_eval("A baseline",   device_topk_baseline)
        d = run_and_eval("D sorted=True", device_topk_sorted)
        # C. fp32 promote: SKIPPED. ttnn.topk asserts input dtype is
        # BFLOAT16 or BFLOAT8_B (topk_device_operation.cpp:148). fp32
        # input would require an upstream change. Recording for the
        # report.
        c = (0, 8, 0.0, 0.0)
        log("  C fp32         : SKIPPED (ttnn.topk requires bf16/bf8_b)")
        b = run_and_eval("B idx_offset", device_topk_idx_offset, idx_offset_tt)

        # Sweep eps magnitudes for B
        log("")
        log("eps sweep for B (idx_offset):")
        for eps in EPS_VARIANTS:
            off_np = (np.arange(NUM_EXPERTS, dtype=np.float32) * eps).reshape(1, 1, NUM_EXPERTS)
            off_tt = _to_tt_replicated(off_np, ttnn, mesh, ttnn.bfloat16)
            offset_scores_tt = ttnn.subtract(scores_biased_tt, off_tt)
            vals_tt, idxs_tt = ttnn.topk(offset_scores_tt, k=TOP_K, dim=-1)
            ttnn.deallocate(offset_scores_tt)
            ttnn.deallocate(off_tt)
            idxs_np = _to_np(idxs_tt, ttnn, mesh).astype(np.int64)
            dev_weights = np.take_along_axis(
                scores_np.reshape(B * S_PADDED, NUM_EXPERTS),
                idxs_np.reshape(B * S_PADDED, TOP_K),
                axis=-1,
            ).reshape(B, S_PADDED, TOP_K)
            denom = dev_weights.sum(axis=-1, keepdims=True) + 1e-20
            dev_weights = dev_weights / denom * ROUTED_SCALING
            evaluate(f"B eps={eps:.0e}", ref_idx, idxs_np, ref_weights, dev_weights)
            ttnn.deallocate(vals_tt)
            ttnn.deallocate(idxs_tt)

        ttnn.deallocate(scores_biased_tt)
        ttnn.deallocate(idx_offset_tt)
        ttnn.deallocate(scores_tt)
        ttnn.deallocate(bias_tt)

        log("")
        log("=" * 70)
        log("REPORT")
        log("=" * 70)
        log(f"  A baseline      : {a[0]}/{a[1]}  cos={a[2]:.6f}")
        log(f"  B idx_offset    : {b[0]}/{b[1]}  cos={b[2]:.6f}")
        log(f"  C fp32_promote  : {c[0]}/{c[1]}  cos={c[2]:.6f}")
        log(f"  D sorted=True   : {d[0]}/{d[1]}  cos={d[2]:.6f}")
        log("")
        # Pass = B is strictly better than A on idx match
        passed = b[0] > a[0] and b[2] >= a[2]
        log(f"v0.4.0h.c topk tie-break {'PASS ✓' if passed else 'FAIL ✗'}")
        return 0 if passed else 1
    finally:
        if own_mesh:
            log("closing mesh…")
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    sys.exit(main())
