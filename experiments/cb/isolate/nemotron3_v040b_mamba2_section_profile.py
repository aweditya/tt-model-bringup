#!/usr/bin/env python3
"""MM7 v0.4.0b — section-by-section profiler for mamba2_block_eager_tt.

v0.4.0a probe surprise: pre-uploading SSD constants only saved 2ms/call.
The wrapper isn't the bottleneck. To target the next refactor we need
to know WHERE the 670ms/layer actually goes inside mamba2_block_eager_tt.

This smoke instruments a single mamba2 layer call with timed sections:
  S1 — rms_norm + in_proj matmul
  S2 — conv1d (device) + readback + reupload (DECODE path)
  S3 — silu + slices
  S4 — readback x/z/B/C/dt to numpy (5 readbacks)
  S5 — SSD step wrapper call
  S6 — y upload + group RMSNorm
  S7 — out_proj matmul + residual add

For the decode path (S=1, state-carried), this isolates which of the
many host bridges around the SSD wrapper dominates.

Output: per-section mean time across 5 calls, sorted by cost.

REUSE: forks the harness-aware pattern from v033b warm-decode-perf.
Uses real layer-0 mamba2 weights via the live State.
"""
from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

N_CALLS = 5
LAYER_IDX = 0  # mamba2 layer; layer_types[0] == "mamba2" in Nemotron-3


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(state=None) -> int:
    os.environ.setdefault("NEMOTRON3_UPLOAD_LAYERS", "all")
    os.environ.setdefault("NEMOTRON3_MOE_MODE", "ep")

    import server_nemotron3_nano_ttnn as srv
    importlib.reload(srv)
    import ttnn

    t_boot = 0.0
    if state is None:
        log("bootstrap…")
        state = srv.State()
        t0 = time.time()
        srv.bootstrap(state, log)
        t_boot = time.time() - t0
        log(f"  bootstrap in {t_boot:.1f}s")
    else:
        log("[harness] reusing live state ✓")

    # Ensure decode-state buffers initialised once (sets ssm/conv state to
    # zeros so mamba2_block_eager_tt takes the "decode" branch via persistent
    # conv_state). Skip the actual reset to avoid clearing live KV caches.
    if (getattr(state, "ssm_state_np", None) is None
            or len(state.ssm_state_np) < srv.N_LAYERS
            or state.ssm_state_np[LAYER_IDX] is None):
        srv.reset_decode_state(state, B=1, log=log)

    HIDDEN = srv.HIDDEN
    layer_idx = LAYER_IDX
    log(f"profiling mamba2 L{layer_idx} (S=1) over {N_CALLS} calls…")

    try:
        # Inline a copy of mamba2_block_eager_tt with section timers.
        # We instrument the DECODE path (persistent conv_state).
        from server_nemotron3_nano_ttnn import (
            EPS, D_INNER, CONV_DIM_M, MAMBA_HEADS, N_GROUPS, SSM_STATE,
            MAMBA_HEAD_DIM, HIFI4, CONV_KERNEL,
        )

        rng = np.random.default_rng(seed=0)
        section_totals = {f"S{i}": 0.0 for i in range(1, 8)}

        def run_one_layer_profile(state, h_input_tt):
            t = {}
            t0 = time.time()
            w = state.per_layer_tt[layer_idx]
            B, S, _ = h_input_tt.shape
            NH = MAMBA_HEADS
            HD = MAMBA_HEAD_DIM
            NG = N_GROUPS
            SS = SSM_STATE

            # ── S1: rms_norm + in_proj ─────────────────────────────
            h_norm_tt = ttnn.rms_norm(h_input_tt, weight=w["norm"], epsilon=EPS)
            in_proj_tt = ttnn.matmul(h_norm_tt, w["in_proj"], compute_kernel_config=HIFI4)
            ttnn.deallocate(h_norm_tt)
            z_tt   = ttnn.slice(in_proj_tt, [0, 0, 0],                    [B, S, D_INNER])
            xBC_tt = ttnn.slice(in_proj_tt, [0, 0, D_INNER],              [B, S, D_INNER + CONV_DIM_M])
            dt_tt  = ttnn.slice(in_proj_tt, [0, 0, D_INNER + CONV_DIM_M], [B, S, D_INNER + CONV_DIM_M + MAMBA_HEADS])
            ttnn.deallocate(in_proj_tt)
            t["S1"] = time.time() - t0

            # ── S2: conv1d (decode path) ───────────────────────────
            t0 = time.time()

            def _rb(tt_):
                arr = ttnn.to_torch(
                    tt_, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
                )
                return arr[:1].float().numpy()

            conv_state = state.conv_state_np[layer_idx]
            xBC_np_S = _rb(xBC_tt)
            combined_np = np.concatenate([conv_state, xBC_np_S], axis=1)
            state.conv_state_np[layer_idx] = combined_np[:, -(CONV_KERNEL - 1):, :].copy()
            combined_tt = ttnn.from_torch(
                torch.from_numpy(combined_np.astype(np.float32)).reshape(
                    B, 1, CONV_KERNEL - 1 + S, CONV_DIM_M,
                ),
                dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=state.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
            )
            conv_full_tt = ttnn.conv1d(
                input_tensor=combined_tt,
                weight_tensor=w["conv1d_w"],
                device=state.mesh,
                in_channels=CONV_DIM_M, out_channels=CONV_DIM_M,
                batch_size=B, input_length=CONV_KERNEL - 1 + S,
                kernel_size=CONV_KERNEL, stride=1,
                padding=0, dilation=1, groups=CONV_DIM_M,
                bias_tensor=w["conv1d_b"],
            )
            ttnn.deallocate(combined_tt)
            conv_full_np = ttnn.to_torch(
                conv_full_tt, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
            )[:1].float().numpy()
            ttnn.deallocate(conv_full_tt)
            if conv_full_np.ndim == 4:
                conv_full_np = conv_full_np.squeeze(1)
            if conv_full_np.shape[-1] != CONV_DIM_M:
                conv_full_np = conv_full_np.transpose(0, 2, 1)
            conv_causal_np = conv_full_np
            conv_causal_tt = ttnn.from_torch(
                torch.from_numpy(conv_causal_np.astype(np.float32)),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
            )
            t["S2"] = time.time() - t0

            # ── S3: silu + slices ──────────────────────────────────
            t0 = time.time()
            silu_out_tt = ttnn.silu(conv_causal_tt)
            ttnn.deallocate(conv_causal_tt)
            BC_SIZE = NG * SS
            x_inner_tt = ttnn.slice(silu_out_tt, [0, 0, 0],                  [B, S, D_INNER])
            B_inner_tt = ttnn.slice(silu_out_tt, [0, 0, D_INNER],            [B, S, D_INNER + BC_SIZE])
            C_inner_tt = ttnn.slice(silu_out_tt, [0, 0, D_INNER + BC_SIZE],  [B, S, D_INNER + 2 * BC_SIZE])
            t["S3"] = time.time() - t0

            # ── S4: readback x/z/B/C/dt to numpy ────────────────────
            t0 = time.time()
            x_inner_np = _rb(x_inner_tt)
            z_full_np  = _rb(z_tt)
            B_inner_np = _rb(B_inner_tt)
            C_inner_np = _rb(C_inner_tt)
            dt_full_np = _rb(dt_tt)
            ttnn.deallocate(x_inner_tt); ttnn.deallocate(B_inner_tt)
            ttnn.deallocate(C_inner_tt); ttnn.deallocate(silu_out_tt)
            ttnn.deallocate(dt_tt)
            x_inner_np = x_inner_np.reshape(B, S, NH, HD)
            z_inner_np = z_full_np.reshape(B, S, NH, HD)
            B_inner_np = B_inner_np.reshape(B, S, NG, SS)
            C_inner_np = C_inner_np.reshape(B, S, NG, SS)
            t["S4"] = time.time() - t0

            # ── S5: SSD step (wrapper) ─────────────────────────────
            t0 = time.time()
            import nemotron3_mamba2_step as _step_mod
            importlib.reload(_step_mod)
            ssm_state = state.ssm_state_np[layer_idx].copy()
            for p in range(S):
                new_state, y_p = _step_mod.mamba2_decode_step_ttnn(
                    x=x_inner_np[:, p, :, :],
                    z=z_inner_np[:, p, :, :],
                    dt=dt_full_np[:, p, :],
                    dt_bias=w["dt_bias_np"], A_log=w["A_log_np"], D=w["D_np"],
                    B_in=B_inner_np[:, p, :, :],
                    C_in=C_inner_np[:, p, :, :],
                    ssm_state=ssm_state,
                    device=state.mesh,
                    debug_mode=5,
                )
                ssm_state = new_state
            state.ssm_state_np[layer_idx] = ssm_state
            y_post_ssd = new_state if S == 1 else None  # placeholder
            y_flat = y_p.reshape(B, S, NH * HD) if S == 1 else None
            t["S5"] = time.time() - t0

            # ── S6: y upload + group RMSNorm ───────────────────────
            t0 = time.time()
            y_tt = ttnn.from_torch(
                torch.from_numpy(y_flat.astype(np.float32)),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
            )
            group_size = D_INNER // NG
            y_grouped = ttnn.reshape(y_tt, [B, S, NG, group_size])
            sq = ttnn.mul(y_grouped, y_grouped)
            var = ttnn.mean(sq, dim=-1, keepdim=True)
            var_eps = ttnn.add(var, EPS)
            rsqrt_var = ttnn.rsqrt(var_eps)
            y_normed_g = ttnn.mul(y_grouped, rsqrt_var)
            y_normed = ttnn.reshape(y_normed_g, [B, S, D_INNER])
            mixer_norm_w_tt = ttnn.from_torch(
                torch.from_numpy(w["mixer_norm_w_np"].astype(np.float32)),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
            )
            y_weighted = ttnn.mul(y_normed, mixer_norm_w_tt)
            silu_z = ttnn.silu(z_tt)
            norm_out_tt = ttnn.mul(y_weighted, silu_z)
            ttnn.deallocate(z_tt); ttnn.deallocate(silu_z); ttnn.deallocate(mixer_norm_w_tt)
            ttnn.deallocate(y_tt); ttnn.deallocate(y_weighted); ttnn.deallocate(y_normed)
            t["S6"] = time.time() - t0

            # ── S7: out_proj + residual ─────────────────────────────
            t0 = time.time()
            o_tt = ttnn.matmul(norm_out_tt, w["out_proj"], compute_kernel_config=HIFI4)
            ttnn.deallocate(norm_out_tt)
            block_tt = ttnn.add(h_input_tt, o_tt)
            ttnn.deallocate(o_tt)
            t["S7"] = time.time() - t0

            return block_tt, t

        for call in range(N_CALLS):
            # Build a random h_input_tt for the layer.
            h_input_np = rng.standard_normal((1, 1, HIDDEN), dtype=np.float32) * 0.1
            h_input_tt = ttnn.from_torch(
                torch.from_numpy(h_input_np),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
            )
            block_tt, t = run_one_layer_profile(state, h_input_tt)
            ttnn.deallocate(h_input_tt)
            ttnn.deallocate(block_tt)
            total = sum(t.values())
            if call == 0:
                log(f"call 0 (cold JIT): total={total*1000:.0f}ms  "
                    f"{ {k: f'{v*1000:.0f}ms' for k,v in t.items()} }")
            else:
                for k, v in t.items():
                    section_totals[k] += v
                if call == N_CALLS - 1:
                    log(f"warm call {call}: total={total*1000:.0f}ms  "
                        f"{ {k: f'{v*1000:.0f}ms' for k,v in t.items()} }")

        log("")
        log("=" * 60)
        log(f"WARM-MEAN SECTIONS (over {N_CALLS - 1} warm calls)")
        log("=" * 60)
        warm_n = N_CALLS - 1
        sorted_sections = sorted(
            [(k, v / warm_n) for k, v in section_totals.items()],
            key=lambda kv: -kv[1],
        )
        total_warm = sum(v for _, v in sorted_sections)
        for k, v in sorted_sections:
            pct = 100.0 * v / total_warm if total_warm > 0 else 0.0
            log(f"  {k}: {v*1000:>6.1f} ms  ({pct:5.1f}%)")
        log(f"  TOTAL warm per layer: {total_warm*1000:.1f} ms")
        log(f"  Projected per-step: 23 layers × {total_warm*1000:.1f} = {23 * total_warm:.2f}s")
        log("")
        log("  → Refactor the top sections first for max impact.")
        return 0
    finally:
        if t_boot > 0:
            log("closing mesh…")
            ttnn.close_mesh_device(state.mesh)


if __name__ == "__main__":
    sys.exit(main())
