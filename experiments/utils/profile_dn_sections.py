#!/usr/bin/env python3
"""Sync-bounded per-section timing of dn_forward_ttnn on (1,4) mesh.

Mirrors dn_forward_ttnn but adds ttnn.synchronize_device + time.perf_counter
around each functional section. Eager mode (matches how the production
server calls dn_forward in non-traced paths).

Why not Tracy: bootstrap floods Tracy's 12000-marker DRAM buffer; device
timings inside the captured region come back as the overflow sentinel
(7,779,xxx μs everywhere). Wall-clock with explicit sync is honest and
sufficient for the "which section is hot" question.

Sections measured per DN forward (same shapes as production):
  1. in_proj_combined matmul + 4 slices  (task 64 fused path)
  2. conv1d update (slice/concat state shift + mul + sum + silu)
  3. q/k/v split + reshape
  4. QK L2 normalize (per Q, per K)
  5. Q scale by 1/sqrt(d_k)
  6. beta + g (qwen36_decay_gate_decode_owned)
  7. GQA repeat (Q/K 4 -> 8 heads)
  8. Recurrence (qwen36_gdn_decode_owned)
  9. RMSNormGated
 10. out_proj
 11. all_reduce

Reports mean / median / std over N iters after warmup. Uses one DN
layer's weights from a freshly bootstrapped State.

Run on qb1:
  cd ~/tt-xla && \\
    TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
    TT_BUILD_DIR=$TT_METAL_HOME/build_Release \\
    ARCH_NAME=blackhole \\
    PYTHONPATH=$TT_METAL_HOME/ttnn \\
    LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
    .venv/bin/python -u experiments/utils/profile_dn_sections.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))
import server_35b_ttnn as srv  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Section:
    """Sync-bounded section timer. Use as `with Section(state, 'name') as s: ...`."""
    def __init__(self, sections_dict, name, mesh):
        self.sections = sections_dict
        self.name = name
        self.mesh = mesh
    def __enter__(self):
        import ttnn
        ttnn.synchronize_device(self.mesh)
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, *a):
        import ttnn
        ttnn.synchronize_device(self.mesh)
        elapsed_ms = (time.perf_counter() - self.t0) * 1000.0
        self.sections.setdefault(self.name, []).append(elapsed_ms)


def dn_forward_instrumented(h_tt, w, mesh, dn_state, sections):
    """Mirror of dn_forward_ttnn with per-section sync-bounded timing.

    Keeps the section names aligned with production (server_35b_ttnn.py
    dn_forward_ttnn). Uses use_owned_gdn=True, use_owned_decay_gate=True.
    """
    import ttnn
    from server_35b_ttnn import (
        HIFI4, CONV_DIM_CHIP, VALUE_DIM_CHIP, NV_PER_CHIP, NK_PER_CHIP,
        HEAD_K_DIM, HEAD_V_DIM, KEY_DIM_CHIP, CONV_KERNEL, EPS,
    )

    conv_state_in, recurrent_state_in = dn_state

    # ── 1. in_proj_combined (task 64 fused) ──
    with Section(sections, "01_in_proj_fused", mesh):
        fused = ttnn.matmul(h_tt, w["in_proj_combined"], compute_kernel_config=HIFI4)
        fr = len(list(fused.shape))
        OFF_QKV_END = CONV_DIM_CHIP
        OFF_Z_END   = OFF_QKV_END + VALUE_DIM_CHIP
        OFF_A_END   = OFF_Z_END + NV_PER_CHIP
        OFF_B_END   = OFF_A_END + NV_PER_CHIP
        if fr == 3:
            mixed_qkv = ttnn.slice(fused, [0, 0, 0],          [1, 1, OFF_QKV_END])
            z         = ttnn.slice(fused, [0, 0, OFF_QKV_END],[1, 1, OFF_Z_END])
            a         = ttnn.slice(fused, [0, 0, OFF_Z_END],  [1, 1, OFF_A_END])
            b         = ttnn.slice(fused, [0, 0, OFF_A_END],  [1, 1, OFF_B_END])
        else:
            mixed_qkv = ttnn.slice(fused, [0, 0],              [1, OFF_QKV_END])
            z         = ttnn.slice(fused, [0, OFF_QKV_END],    [1, OFF_Z_END])
            a         = ttnn.slice(fused, [0, OFF_Z_END],      [1, OFF_A_END])
            b         = ttnn.slice(fused, [0, OFF_A_END],      [1, OFF_B_END])
        ttnn.deallocate(fused)

    # ── 2. conv1d update + silu ──
    with Section(sections, "02_conv1d_silu", mesh):
        cs_rank = len(list(conv_state_in.shape))
        cur = ttnn.reshape(mixed_qkv, [1, CONV_DIM_CHIP, 1])
        ttnn.deallocate(mixed_qkv)
        if cs_rank == 4:
            prior = ttnn.slice(conv_state_in, [0, 0, 0, 1], [1, 1, CONV_DIM_CHIP, CONV_KERNEL])
            prior = ttnn.reshape(prior, [1, CONV_DIM_CHIP, CONV_KERNEL - 1])
        else:
            prior = ttnn.slice(conv_state_in, [0, 0, 1], [1, CONV_DIM_CHIP, CONV_KERNEL])
        conv_state_new = ttnn.concat([prior, cur], dim=-1)
        ttnn.deallocate(prior); ttnn.deallocate(cur)
        cw_rank_local = len(list(w["conv1d_weight"].shape))
        if cw_rank_local == 4:
            w_conv = ttnn.reshape(w["conv1d_weight"], [1, CONV_DIM_CHIP, CONV_KERNEL])
        else:
            w_conv = w["conv1d_weight"]
        state_w = ttnn.mul(conv_state_new, w_conv)
        if cw_rank_local == 4:
            ttnn.deallocate(w_conv)
        conv_out_3d = ttnn.sum(state_w, dim=-1, keepdim=True)
        ttnn.deallocate(state_w)
        conv_out = ttnn.reshape(conv_out_3d, [1, CONV_DIM_CHIP])
        ttnn.deallocate(conv_out_3d)
        silu_out = ttnn.silu(conv_out)
        ttnn.deallocate(conv_out)

    # ── 3. q/k/v split + reshape ──
    with Section(sections, "03_qkv_split", mesh):
        sr = len(list(silu_out.shape))
        if sr == 3:
            q_flat = ttnn.slice(silu_out, [0, 0, 0], [1, 1, KEY_DIM_CHIP])
            k_flat = ttnn.slice(silu_out, [0, 0, KEY_DIM_CHIP], [1, 1, 2 * KEY_DIM_CHIP])
            v_flat = ttnn.slice(silu_out, [0, 0, 2 * KEY_DIM_CHIP], [1, 1, CONV_DIM_CHIP])
        else:
            q_flat = ttnn.slice(silu_out, [0, 0], [1, KEY_DIM_CHIP])
            k_flat = ttnn.slice(silu_out, [0, KEY_DIM_CHIP], [1, 2 * KEY_DIM_CHIP])
            v_flat = ttnn.slice(silu_out, [0, 2 * KEY_DIM_CHIP], [1, CONV_DIM_CHIP])
        ttnn.deallocate(silu_out)
        q_h = ttnn.reshape(q_flat, [1, NK_PER_CHIP, HEAD_K_DIM])
        k_h = ttnn.reshape(k_flat, [1, NK_PER_CHIP, HEAD_K_DIM])
        v_h = ttnn.reshape(v_flat, [1, NV_PER_CHIP, HEAD_V_DIM])

    # ── 4. QK L2 normalize (manual chain — main candidate for rms_norm fusion) ──
    with Section(sections, "04_qk_l2norm", mesh):
        q_sq = ttnn.mul(q_h, q_h)
        q_sumsq = ttnn.sum(q_sq, dim=-1, keepdim=True)
        ttnn.deallocate(q_sq)
        q_inv = ttnn.rsqrt(ttnn.add(q_sumsq, EPS))
        ttnn.deallocate(q_sumsq)
        q_n = ttnn.mul(q_h, q_inv)
        ttnn.deallocate(q_h); ttnn.deallocate(q_inv)

        k_sq = ttnn.mul(k_h, k_h)
        k_sumsq = ttnn.sum(k_sq, dim=-1, keepdim=True)
        ttnn.deallocate(k_sq)
        k_inv = ttnn.rsqrt(ttnn.add(k_sumsq, EPS))
        ttnn.deallocate(k_sumsq)
        k_n = ttnn.mul(k_h, k_inv)
        ttnn.deallocate(k_h); ttnn.deallocate(k_inv)

    # ── 5. Q scale by 1/sqrt(d_k) ──
    with Section(sections, "05_q_scale", mesh):
        q_scale = 1.0 / (HEAD_K_DIM ** 0.5)
        q_n_scaled = ttnn.multiply(q_n, q_scale)
        ttnn.deallocate(q_n)
        q_n = q_n_scaled
    q_h = q_n; k_h = k_n

    # ── 6. beta + g (owned decay_gate) ──
    with Section(sections, "06_decay_gate", mesh):
        a_r2 = ttnn.reshape(a, [1, NV_PER_CHIP])
        b_r2 = ttnn.reshape(b, [1, NV_PER_CHIP])
        dt_bias_r2 = ttnn.reshape(w["dt_bias"], [1, NV_PER_CHIP])
        A_log_r2 = ttnn.reshape(w["A_log"], [1, NV_PER_CHIP])
        g_decay, beta = ttnn.experimental.qwen36_decay_gate_decode_owned(
            a_r2, b_r2, dt_bias_r2, A_log_r2)
        ttnn.deallocate(a_r2); ttnn.deallocate(b_r2)
        ttnn.deallocate(a); ttnn.deallocate(b)

    # ── 7. GQA repeat (Q, K 4 -> 8 heads) ──
    with Section(sections, "07_gqa_repeat", mesh):
        GQA_REPEAT = NV_PER_CHIP // NK_PER_CHIP
        q_4d = ttnn.reshape(q_h, [1, NK_PER_CHIP, 1, HEAD_K_DIM])
        k_4d = ttnn.reshape(k_h, [1, NK_PER_CHIP, 1, HEAD_K_DIM])
        ttnn.deallocate(q_h); ttnn.deallocate(k_h)
        q_rep_4d = ttnn.repeat(q_4d, ttnn.Shape([1, 1, GQA_REPEAT, 1]))
        k_rep_4d = ttnn.repeat(k_4d, ttnn.Shape([1, 1, GQA_REPEAT, 1]))
        ttnn.deallocate(q_4d); ttnn.deallocate(k_4d)
        q_rep = ttnn.reshape(q_rep_4d, [1, NV_PER_CHIP, HEAD_K_DIM])
        k_rep = ttnn.reshape(k_rep_4d, [1, NV_PER_CHIP, HEAD_K_DIM])
        ttnn.deallocate(q_rep_4d); ttnn.deallocate(k_rep_4d)

    # ── 8. Recurrence (owned_gdn) ──
    with Section(sections, "08_owned_gdn", mesh):
        alpha = ttnn.reshape(g_decay, [1, NV_PER_CHIP, 1, 1])
        beta_r = ttnn.reshape(beta, [1, NV_PER_CHIP, 1, 1])
        ttnn.deallocate(g_decay); ttnn.deallocate(beta)
        q_4d = ttnn.reshape(q_rep, [1, NV_PER_CHIP, 1, HEAD_K_DIM])
        k_4d = ttnn.reshape(k_rep, [1, NV_PER_CHIP, 1, HEAD_K_DIM])
        v_4d = ttnn.reshape(v_h, [1, NV_PER_CHIP, 1, HEAD_V_DIM])
        ttnn.deallocate(q_rep); ttnn.deallocate(k_rep); ttnn.deallocate(v_h)
        state_5d_clone = ttnn.add(recurrent_state_in, 0.0)
        H_owned_in = ttnn.reshape(state_5d_clone, [1, NV_PER_CHIP, HEAD_K_DIM, HEAD_V_DIM])
        out_4d, H_owned_out = ttnn.experimental.qwen36_gdn_decode_owned(
            H_owned_in, q_4d, k_4d, v_4d, alpha, beta_r,
        )
        ttnn.deallocate(q_4d); ttnn.deallocate(k_4d); ttnn.deallocate(v_4d)
        ttnn.deallocate(alpha); ttnn.deallocate(beta_r)
        ttnn.deallocate(H_owned_in)
        new_recurrent_state = ttnn.reshape(H_owned_out, recurrent_state_in.shape)
        ttnn.copy(new_recurrent_state, recurrent_state_in)
        ttnn.deallocate(H_owned_out); ttnn.deallocate(new_recurrent_state)
        ttnn.deallocate(state_5d_clone)

    # ── 9. RMSNormGated ──
    with Section(sections, "09_rms_norm_gated", mesh):
        out_3d = ttnn.reshape(out_4d, [1, NV_PER_CHIP, HEAD_V_DIM])
        ttnn.deallocate(out_4d)
        # RMSNorm per head
        out_sq = ttnn.mul(out_3d, out_3d)
        out_meansq = ttnn.mean(out_sq, dim=-1, keepdim=True)
        ttnn.deallocate(out_sq)
        out_inv = ttnn.rsqrt(ttnn.add(out_meansq, EPS))
        ttnn.deallocate(out_meansq)
        out_normed = ttnn.mul(out_3d, out_inv)
        ttnn.deallocate(out_3d); ttnn.deallocate(out_inv)
        nw_rank = len(list(w["norm_weight"].shape))
        if nw_rank == 3:
            nw = w["norm_weight"]
        else:
            nw = ttnn.reshape(w["norm_weight"], [1, 1, HEAD_V_DIM])
        out_weighted = ttnn.mul(out_normed, nw)
        if nw_rank != 3:
            ttnn.deallocate(nw)
        ttnn.deallocate(out_normed)
        # Gate by silu(z)
        z_h = ttnn.reshape(z, [1, NV_PER_CHIP, HEAD_V_DIM])
        ttnn.deallocate(z)
        z_silu = ttnn.silu(z_h)
        ttnn.deallocate(z_h)
        out_gated = ttnn.mul(out_weighted, z_silu)
        ttnn.deallocate(out_weighted); ttnn.deallocate(z_silu)

    # ── 10. out_proj + 11. all_reduce ──
    with Section(sections, "10_out_proj", mesh):
        out_flat = ttnn.reshape(out_gated, [1, NV_PER_CHIP * HEAD_V_DIM])
        ttnn.deallocate(out_gated)
        proj = ttnn.matmul(out_flat, w["out_proj"], compute_kernel_config=HIFI4)
        ttnn.deallocate(out_flat)

    import ttnn
    with Section(sections, "11_all_reduce", mesh):
        from ttnn import CoreGrid
        all_reduced = ttnn.experimental.all_reduce_async(
            proj, num_links=2, topology=ttnn.Topology.Linear,
        )
        ttnn.deallocate(proj)
        ttnn.synchronize_device(mesh)

    # Final state mutation: conv_state_new -> conv_state_in (in-place copy)
    ttnn.copy(conv_state_new, conv_state_in)
    ttnn.deallocate(conv_state_new)

    return all_reduced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-warmup", type=int, default=3)
    ap.add_argument("--n-iters", type=int, default=20)
    ap.add_argument("--layer-idx", type=int, default=0)
    args = ap.parse_args()

    log("bootstrap…")
    state = srv.State()
    state.moe_mode = "pattern_a_batched"
    srv.bootstrap(state, log)
    state.reset_caches_ttnn()

    log("synthesizing h…")
    rng = np.random.default_rng(0)
    h_np = rng.normal(0, 5.0, size=(1, srv.HIDDEN)).astype(np.float32)
    h_tt = srv.np_to_replicated(h_np, state.mesh)

    layer_idx = args.layer_idx
    assert state.layer_types[layer_idx] == "linear_attention"
    w = state.per_layer_tt[layer_idx]
    dn_state = state.dn_caches_tt[layer_idx]
    log(f"layer {layer_idx} ready (linear_attention)")

    sections = {}
    import ttnn
    log(f"warmup x{args.n_warmup}…")
    for _ in range(args.n_warmup):
        out = dn_forward_instrumented(h_tt, w, state.mesh, dn_state, {})
        ttnn.deallocate(out)
    ttnn.synchronize_device(state.mesh)

    log(f"timed x{args.n_iters}…")
    for _ in range(args.n_iters):
        out = dn_forward_instrumented(h_tt, w, state.mesh, dn_state, sections)
        ttnn.deallocate(out)

    log("\n=== per-section means (ms) ===")
    total_mean = sum(np.mean(v) for v in sections.values())
    for name in sorted(sections):
        ts = np.array(sections[name])
        log(f"  {name:25s} mean {ts.mean():7.3f}  median {np.median(ts):7.3f}  "
            f"std {ts.std():6.3f}  ({100*ts.mean()/total_mean:5.1f}%)")
    log(f"  {'TOTAL (sum-of-sections)':25s} mean {total_mean:7.3f} ms")
    log(f"  per-token (x30 DN layers) -> {total_mean * 30:.1f} ms/tok")

    ttnn.deallocate(h_tt)
    ttnn.close_mesh_device(state.mesh)


if __name__ == "__main__":
    main()
