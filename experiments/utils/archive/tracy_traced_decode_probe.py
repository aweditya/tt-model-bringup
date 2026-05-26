#!/usr/bin/env python3
"""
Tracy traced-decode op-time probe for Qwen3.6-27B single-chip.

Goal: produce per-op device-side timing for a captured-then-executed decode
trace, so we can identify which op categories (matmul / rms_norm / slice /
reshape / concat / SDPA / paged_update_cache / etc.) dominate per-step time.

This is the qb1 fallback for the multi-chip TP profiling task. The 4-chip TP
trace runs ~142 ms/tok and we want to understand where the "missing" 77 ms vs
the 4× scaling target of 50 ms/tok comes from. qb2 lacks a Tracy-enabled
tt-metal build (cmake/clang prereqs absent), so we measure the single-chip
traced step (which has the same per-op categories minus collectives) and
extrapolate.

Run pattern (on qb1):
    cd ~/tt-xla
    bash experiments/utils/run_with_tracy_build.sh \\
        ~/tt-xla/.venv/bin/python -m tracy -r -p \\
        --device-trace-profiler \\
        -o research/probe_logs/tracy_qb1_traced \\
        experiments/utils/tracy_traced_decode_probe.py

The `-r` flag generates `ops_perf_results_*.csv` under the output folder; `-p`
keeps only zones we explicitly start/stop; `--device-trace-profiler` enables
TT_METAL_TRACE_PROFILER=1 which captures device timings INSIDE the trace
(otherwise only host launch times are recorded).

The script:
  1. Opens device 0
  2. Loads ONE Qwen3.6-27B linear_attention (layer 0) and ONE full_attention
     (layer 3) weight bundle. Reuses these as N=8 chained layers
     (6 DN + 2 attn) to mirror the production layer-type ratio.
  3. Pre-allocates state buffers (ssm/conv per DN, kv_cache per attn) so
     forward_step reads from fixed addresses (trace-safe).
  4. Warmup 2× eager forward (JIT all kernels — required pre-capture, per
     feedback_c4v4_validated.md).
  5. begin_trace_capture → forward_one_step → end_trace_capture
  6. execute_trace × 5 with sync between each.

Tracy/process_ops_logs.py then dumps a CSV with per-op DEVICE KERNEL DURATION.
"""
import os
import sys
import time
import json
from collections import defaultdict

import numpy as np
import torch
import ttnn
from huggingface_hub import hf_hub_download

sys.stdout.reconfigure(line_buffering=True)
PROJECT_ROOT = os.path.expanduser("~/tt-xla")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments"))

# Import the on-device forward primitives from 91f via importlib because
# the filename starts with a digit (91f_…) which isn't a valid Python module
# name.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_module_91f",
    os.path.join(PROJECT_ROOT, "experiments", "91f_qwen36_27b_full_ondevice.py"),
)
_m91f = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m91f)
deltanet_step_ondevice_traced = _m91f.deltanet_step_ondevice_traced
gated_attn_step_ondevice_traced = _m91f.gated_attn_step_ondevice_traced
mlp_step_ondevice = _m91f.mlp_step_ondevice
load_layer_weights_all = _m91f.load_layer_weights_all
upload = _m91f.upload
EPS = _m91f.EPS
MODEL_ID = _m91f.MODEL_ID


# --- config -----------------------------------------------------------------
MAX_POS = 128
N_LAYERS_DN = 6
N_LAYERS_ATTN = 2
N_TRACE_EXEC = 5
N_WARMUP = 2


def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    print("=" * 78)
    print("Tracy traced-decode probe — Qwen3.6-27B single-chip (qb1)")
    print(f"timestamp: {ts}")
    print(f"chain: {N_LAYERS_DN} DN + {N_LAYERS_ATTN} attn = {N_LAYERS_DN + N_LAYERS_ATTN} layers")
    print("=" * 78)

    # --- HF config --------------------------------------------------------
    print("\n[1/8] Load HF config…")
    cfg_path = hf_hub_download(MODEL_ID, "config.json")
    with open(cfg_path) as f:
        text_cfg = json.load(f)["text_config"]
    cfg = {
        "hidden":      text_cfg["hidden_size"],
        "n_k_heads":   text_cfg["linear_num_key_heads"],
        "n_v_heads":   text_cfg["linear_num_value_heads"],
        "k_dim":       text_cfg["linear_key_head_dim"],
        "v_dim":       text_cfg["linear_value_head_dim"],
        "conv_kernel": text_cfg["linear_conv_kernel_dim"],
        "n_q_heads":   text_cfg["num_attention_heads"],
        "n_kv_heads":  text_cfg["num_key_value_heads"],
        "head_dim":    text_cfg["head_dim"],
        "partial_rotary_factor": text_cfg["partial_rotary_factor"],
    }
    HIDDEN = cfg["hidden"]
    KEY_DIM = cfg["n_k_heads"] * cfg["k_dim"]
    VAL_DIM = cfg["n_v_heads"] * cfg["v_dim"]
    CONV_DIM = 2 * KEY_DIM + VAL_DIM
    print(f"  HIDDEN={HIDDEN}  N_Q={cfg['n_q_heads']}  N_KV={cfg['n_kv_heads']}  HEAD_DIM={cfg['head_dim']}")
    print(f"  KEY_DIM={KEY_DIM}  VAL_DIM={VAL_DIM}  CONV_DIM={CONV_DIM}")

    # --- device ----------------------------------------------------------
    print("\n[2/8] Open device 0…")
    device = ttnn.open_device(device_id=0)

    try:
        # --- load weights ------------------------------------------------
        print("\n[3/8] Load 1 linear_attention + 1 full_attention layer weight bundles…")
        w_dn_np = load_layer_weights_all(0, "linear_attention")
        w_attn_np = load_layer_weights_all(3, "full_attention")

        def upload_layer(w_np):
            w_tt = {}
            for k, arr in w_np.items():
                if k == "conv1d_weight" and arr.ndim == 3:
                    arr = arr.squeeze(1)
                w_tt[k] = upload(arr, device, dtype=ttnn.bfloat16)
            return w_tt

        w_dn_tt = upload_layer(w_dn_np)
        w_attn_tt = upload_layer(w_attn_np)
        ttnn.synchronize_device(device)
        print(f"  ✓ DN keys: {len(w_dn_tt)}; attn keys: {len(w_attn_tt)}")

        # --- pre-allocate state buffers ---------------------------------
        print("\n[4/8] Pre-allocate per-layer state buffers + scalars…")
        ssm_states = [
            upload(np.zeros((cfg["n_v_heads"], cfg["k_dim"], cfg["v_dim"]),
                            dtype=np.float32),
                   device, dtype=ttnn.float32)
            for _ in range(N_LAYERS_DN)
        ]
        conv_states = [
            upload(np.zeros((CONV_DIM, cfg["conv_kernel"] - 1), dtype=np.float32),
                   device, dtype=ttnn.bfloat16)
            for _ in range(N_LAYERS_DN)
        ]

        kv_k_list = []
        kv_v_list = []
        for _ in range(N_LAYERS_ATTN):
            kv_k_np = np.zeros((1, cfg["n_kv_heads"], MAX_POS, cfg["head_dim"]),
                               dtype=np.float32)
            kv_v_np = np.zeros((1, cfg["n_kv_heads"], MAX_POS, cfg["head_dim"]),
                               dtype=np.float32)
            kv_k_list.append(ttnn.from_torch(
                torch.from_numpy(kv_k_np), dtype=ttnn.bfloat16,
                device=device, layout=ttnn.TILE_LAYOUT))
            kv_v_list.append(ttnn.from_torch(
                torch.from_numpy(kv_v_np), dtype=ttnn.bfloat16,
                device=device, layout=ttnn.TILE_LAYOUT))

        # Position + RoPE tables (one shared set across all attn layers)
        cur_pos = 0
        cur_pos_tt = ttnn.from_torch(
            torch.tensor([cur_pos], dtype=torch.int32), device=device)

        rotary_dim = int(cfg["head_dim"] * cfg["partial_rotary_factor"])
        half_rot = rotary_dim // 2
        freqs = 1.0 / (10_000_000.0 ** (np.arange(half_rot).astype(np.float32) / half_rot))
        angles = cur_pos * freqs
        cos_np = np.concatenate([np.cos(angles), np.cos(angles)]).astype(np.float32)
        sin_np = np.concatenate([np.sin(angles), np.sin(angles)]).astype(np.float32)
        cos_tt = upload(cos_np, device, dtype=ttnn.bfloat16)
        sin_tt = upload(sin_np, device, dtype=ttnn.bfloat16)

        # Scatter index for gated_attn_step_ondevice_traced — pre-built per-step
        # index tensor [1, N_KV, 1, HEAD_DIM] of int32 set to cur_pos
        idx_np = np.full((1, cfg["n_kv_heads"], 1, cfg["head_dim"]), cur_pos, dtype=np.int32)
        index_tt = ttnn.from_torch(
            torch.from_numpy(idx_np), dtype=ttnn.int32,
            device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

        # Input buffer
        x_init_np = np.random.RandomState(42).randn(HIDDEN).astype(np.float32) * 0.5
        x_buf = upload(x_init_np.reshape(1, HIDDEN), device, dtype=ttnn.bfloat16)

        print(f"  ✓ {N_LAYERS_DN} ssm/conv + {N_LAYERS_ATTN} kv buffers")

        # --- forward chain ------------------------------------------------
        def forward_one_step():
            """One chained forward — N_LAYERS_DN DN blocks + N_LAYERS_ATTN attn blocks,
            each followed by MLP. Mirrors production layer structure."""
            x_tt = x_buf
            attn_idx = 0
            dn_idx = 0
            # Interleave: roughly 3 DN per 1 attn (production has 48:16 ratio = 3:1)
            sequence = []
            for i in range(N_LAYERS_DN + N_LAYERS_ATTN):
                if i % 4 != 3 and dn_idx < N_LAYERS_DN:
                    sequence.append("dn")
                elif attn_idx < N_LAYERS_ATTN:
                    sequence.append("attn")
                else:
                    sequence.append("dn")
            for layer_type in sequence:
                if layer_type == "dn":
                    x_tt = deltanet_step_ondevice_traced(
                        x_tt, w_dn_tt, ssm_states[dn_idx], conv_states[dn_idx], cfg)
                    x_tt = mlp_step_ondevice(x_tt, w_dn_tt)
                    dn_idx += 1
                else:
                    x_tt = gated_attn_step_ondevice_traced(
                        x_tt, w_attn_tt, kv_k_list[attn_idx], kv_v_list[attn_idx],
                        cur_pos_tt, cos_tt, sin_tt, index_tt, cfg)
                    x_tt = mlp_step_ondevice(x_tt, w_attn_tt)
                    attn_idx += 1
            return x_tt

        # --- warmup eager ------------------------------------------------
        print(f"\n[5/8] Warmup eager × {N_WARMUP} (JIT amortization)…")
        for i in range(N_WARMUP):
            t0 = time.perf_counter()
            _ = forward_one_step()
            ttnn.synchronize_device(device)
            print(f"  warmup {i}: {(time.perf_counter()-t0)*1000:.1f} ms")

        # --- capture trace ------------------------------------------------
        print("\n[6/8] begin_trace_capture + forward_one_step + end_trace_capture…")
        t_cap = time.perf_counter()
        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        _ = forward_one_step()
        ttnn.end_trace_capture(device, trace_id, cq_id=0)
        ttnn.synchronize_device(device)
        print(f"  ✓ trace captured in {(time.perf_counter()-t_cap)*1000:.1f} ms")

        # --- execute trace × N -------------------------------------------
        print(f"\n[7/8] execute_trace × {N_TRACE_EXEC} (with Tracy zones)…")
        exec_times_ms = []
        for i in range(N_TRACE_EXEC):
            ttnn.synchronize_device(device)
            ttnn.start_tracy_zone(
                "tracy_traced_decode_probe.py",
                f"execute_trace_step_{i}",
                0,  # line number
            )
            t0 = time.perf_counter()
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            t_ms = (time.perf_counter() - t0) * 1000
            ttnn.stop_tracy_zone(f"execute_trace_step_{i}", 0)
            exec_times_ms.append(t_ms)
            print(f"  step {i}: {t_ms:.2f} ms")
        print(f"\n  median = {float(np.median(exec_times_ms)):.2f} ms")
        print(f"  mean   = {float(np.mean(exec_times_ms)):.2f} ms")
        print(f"  per-layer median = {float(np.median(exec_times_ms))/(N_LAYERS_DN+N_LAYERS_ATTN):.2f} ms")

        # --- release trace ------------------------------------------------
        print("\n[8/8] release_trace + close device…")
        ttnn.release_trace(device, trace_id)
    finally:
        ttnn.close_device(device)
        print("  ✓ device closed")


if __name__ == "__main__":
    main()
