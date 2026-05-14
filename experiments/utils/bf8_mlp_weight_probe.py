#!/usr/bin/env python3
"""Probe: bf8 vs bf16 MLP projection weights on Qwen3.6-27B (qb1 device 3).

Premise check (read 2026-05-13 from production code):
  - experiments/serve/server.py:153-154 ALREADY loads MLP gate/up/down as
    ttnn.bfloat8_b. Same in experiments/utils/perf_baseline.py:349-350.
  - experiments/91f_qwen36_27b_full_ondevice.py:697 (the standalone bench)
    uploads as bf16 — but that path is a 4-layer cosine sanity check, NOT
    the production decode loop the user sees.
  - The persistent server's ~200 ms/tok number is therefore ALREADY with
    bf8 MLP weights. The honest experiment is the ABLATION in the other
    direction: how much quality/latency would going BACK to bf16 cost?

This probe runs ONE MLP step (gate→silu, up, mul, down + residual) at
production Qwen3.6-27B shape with REAL layer-0 weights, with three variants:
  V_bf16:    gate/up/down all bfloat16  (size = 1.0 × bf8 reference; max quality)
  V_bf8:     gate/up/down all bfloat8_b (current production)
  V_mixed:   gate/up at bf16, down at bf8 (sometimes used in TT designs to
             reduce down_proj round-trip variance while keeping cheap up/gate)

Compares:
  - cosine of each variant vs an fp32 numpy oracle MLP (silu, eltwise mul,
    eltwise add of residual).
  - cosine of V_bf16 vs V_bf8 (rounding-profile delta).
  - sync-bounded per-step latency.

API check: bf8 matmul on Blackhole is ALREADY validated in production by
  experiments/utils/dn_fusion_isolation_probe.py:94-104 (4 bf8 ttnn.linears)
  experiments/utils/kernel_profile_probe.py:43,69 (w_dtype=ttnn.bfloat8_b)
  experiments/utils/perf_baseline.py:349-350 (bf8 for all proj in 64 layers)
This probe is the ISOLATION ablation; no fresh API smoke needed.

Shapes (Qwen3.6-27B text MLP):
  HIDDEN = 5120, INTERMEDIATE = 25600
  gate_proj.weight: [5120, 25600]  (transposed at load time in 91f)
  up_proj.weight:   [5120, 25600]
  down_proj.weight: [25600, 5120]

NOT to touch (per non-negotiables 5/N4):
  - device 0 (persistent server PID 705445)
  - device 1 (Agent J)
  - device 2 (Agent K)

Run:
  ssh qb1 'cd ~/tt-xla && .venv/bin/python -m experiments.utils.bf8_mlp_weight_probe'
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import importlib.util
import os
import time
import traceback

import numpy as np
import torch
import ttnn


# Device selection: with TT_VISIBLE_DEVICES=3 masking out chips 0/1/2 (held by
# server + other agents), the only visible chip becomes local id 0. Without
# masking, opening any device_id triggers UMD cluster-wide enumeration which
# blocks on the server's chip-0 lock. We therefore default to local-id 0 and
# rely on the caller setting TT_VISIBLE_DEVICES=3 in the shell.
DEVICE_ID = int(os.environ.get("PROBE_DEVICE_ID", "0"))
N_WARMUP = 5
N_ITER = 50
N_INPUTS = 4
HIDDEN = 5120
INTERMEDIATE = 25600
SEED = 0xBF8FED
EPS = 1e-6

_91F_PATH = os.path.expanduser("~/tt-xla/experiments/91f_qwen36_27b_full_ondevice.py")


def load_91f():
    spec = importlib.util.spec_from_file_location("_91f_mod", _91F_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def silu_np(x):
    return x * (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)


def numpy_mlp_oracle(x_np, gate_np, up_np, down_np, in_norm_np):
    """fp32 reference MLP with RMS-norm + residual, matching mlp_step_ondevice math.

    91f.mlp_step_ondevice:
       h = rms_norm(x, weight=post_attention_layernorm)
       g = silu(h @ gate_proj)
       u = h @ up_proj
       out = (g * u) @ down_proj
       return x + out
    """
    x = x_np.astype(np.float64)
    rms = np.sqrt(np.mean(x ** 2) + EPS)
    h = (x / rms) * in_norm_np.astype(np.float64)
    h = h.astype(np.float32)
    g = silu_np(h @ gate_np.astype(np.float64))
    u = (h @ up_np.astype(np.float64)).astype(np.float32)
    out = ((g * u).astype(np.float64) @ down_np.astype(np.float64)).astype(np.float32)
    return (x.astype(np.float32) + out)


def upload_weight(arr, device, dtype):
    t = torch.from_numpy(np.ascontiguousarray(arr.astype(np.float32)))
    while t.dim() < 2:
        t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)


def cosine(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def build_w_tt(mod, weights_np, device, proj_dtype, norm_dtype):
    """Mirror 91f weight upload for an MLP-only weight dict."""
    w_tt = {}
    for k in ("gate_proj", "up_proj", "down_proj"):
        w_tt[k] = upload_weight(weights_np[k], device, dtype=proj_dtype)
    w_tt["post_attention_layernorm"] = upload_weight(
        weights_np["post_attention_layernorm"], device, dtype=norm_dtype)
    return w_tt


def run_variant(mod, label, weights_np, x_full_np, device, gate_dtype, up_dtype,
                down_dtype):
    """Build w_tt with per-weight dtypes; bench latency; return outputs for cosine."""
    # Build weight dict matching mlp_step_ondevice expectations.
    # We allow per-key dtype to support the mixed variant.
    w_tt = {}
    w_tt["gate_proj"] = upload_weight(weights_np["gate_proj"], device, dtype=gate_dtype)
    w_tt["up_proj"]   = upload_weight(weights_np["up_proj"],   device, dtype=up_dtype)
    w_tt["down_proj"] = upload_weight(weights_np["down_proj"], device, dtype=down_dtype)
    w_tt["post_attention_layernorm"] = upload_weight(
        weights_np["post_attention_layernorm"], device, dtype=ttnn.bfloat16)

    # Warmup
    for w_i in range(N_WARMUP):
        x_tt = upload_weight(x_full_np[w_i % N_INPUTS].reshape(1, HIDDEN),
                              device, dtype=ttnn.bfloat16)
        _ = mod.mlp_step_ondevice(x_tt, w_tt)
    ttnn.synchronize_device(device)

    # Timed
    latencies_ms = []
    outputs = []
    for it in range(N_ITER):
        x_np = x_full_np[it % N_INPUTS]
        x_tt = upload_weight(x_np.reshape(1, HIDDEN), device, dtype=ttnn.bfloat16)
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        out_tt = mod.mlp_step_ondevice(x_tt, w_tt)
        ttnn.synchronize_device(device)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        if it < N_INPUTS:
            out_np = ttnn.to_torch(out_tt).float().cpu().numpy().flatten()[:HIDDEN]
            outputs.append(out_np)

    median_ms = float(np.median(latencies_ms))
    mean_ms   = float(np.mean(latencies_ms))
    p95_ms    = float(np.percentile(latencies_ms, 95))

    # Free big weights before returning so the next variant has room.
    for k in list(w_tt.keys()):
        ttnn.deallocate(w_tt[k])

    print(f"  [{label}] median={median_ms:.3f} ms  mean={mean_ms:.3f} ms  "
          f"p95={p95_ms:.3f} ms")
    return median_ms, mean_ms, p95_ms, outputs


def main():
    print("=" * 70)
    print(f"bf8 MLP weight probe (qb1, device {DEVICE_ID}, Qwen3.6-27B layer 0)")
    print(f"  HIDDEN={HIDDEN}  INTERMEDIATE={INTERMEDIATE}")
    print(f"  N_WARMUP={N_WARMUP}  N_ITER={N_ITER}  N_INPUTS={N_INPUTS}")
    print("=" * 70)

    mod = load_91f()
    print("[1/4] Loaded 91f module")

    print("[2/4] Loading real Qwen3.6-27B layer-0 MLP weights from HF…")
    t0 = time.time()
    w_np = mod.load_layer_weights_all(0, "linear_attention")
    print(f"      gate_proj   {w_np['gate_proj'].shape}  dtype={w_np['gate_proj'].dtype}")
    print(f"      up_proj     {w_np['up_proj'].shape}")
    print(f"      down_proj   {w_np['down_proj'].shape}")
    print(f"      post_attn_ln {w_np['post_attention_layernorm'].shape}  (already 1+w)")
    print(f"      load time:  {time.time()-t0:.1f}s")

    rng = np.random.default_rng(SEED)
    x_full_np = (rng.standard_normal((N_INPUTS, HIDDEN)) * 0.1).astype(np.float32)

    # fp32 numpy oracle
    print("[3/4] Computing numpy fp32 oracle MLP for cosine baseline…")
    oracle_outputs = [
        numpy_mlp_oracle(x_full_np[i], w_np["gate_proj"], w_np["up_proj"],
                          w_np["down_proj"], w_np["post_attention_layernorm"])
        for i in range(N_INPUTS)
    ]
    print(f"      computed {N_INPUTS} oracle outputs")

    print(f"[4/4] Opening device {DEVICE_ID} and running variants…")
    device = ttnn.open_device(device_id=DEVICE_ID)
    try:
        variants = [
            ("V_bf16  (all bf16)",     ttnn.bfloat16, ttnn.bfloat16, ttnn.bfloat16),
            ("V_bf8   (all bf8 [PROD])", ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b),
            ("V_mixed (gate/up bf16, down bf8)",
                ttnn.bfloat16, ttnn.bfloat16, ttnn.bfloat8_b),
        ]
        all_results = {}
        for label, gd, ud, dd in variants:
            print(f"\n  Running {label}…")
            try:
                res = run_variant(mod, label, w_np, x_full_np, device, gd, ud, dd)
                all_results[label] = res
            except Exception as e:
                print(f"  [{label}] FAILED: {e}")
                traceback.print_exc()

        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        for label, (median, mean, p95, outs) in all_results.items():
            cos_vs_oracle = [cosine(oracle_outputs[i], outs[i]) for i in range(N_INPUTS)]
            avg_cos = float(np.mean(cos_vs_oracle))
            min_cos = float(np.min(cos_vs_oracle))
            maxd = float(np.max([np.max(np.abs(oracle_outputs[i] - outs[i]))
                                  for i in range(N_INPUTS)]))
            print(f"\n  {label}:")
            print(f"    latency: median={median:.3f} ms  mean={mean:.3f} ms  p95={p95:.3f} ms")
            print(f"    cos vs fp32 oracle: avg={avg_cos:.6f}  min={min_cos:.6f}  "
                  f"max|Δ|={maxd:.4e}")
            print(f"    per-input cos: {['%.6f' % c for c in cos_vs_oracle]}")

        # Cross-variant rounding-profile delta
        if "V_bf16  (all bf16)" in all_results and "V_bf8   (all bf8 [PROD])" in all_results:
            bf16_outs = all_results["V_bf16  (all bf16)"][3]
            bf8_outs  = all_results["V_bf8   (all bf8 [PROD])"][3]
            cross_cos = [cosine(bf16_outs[i], bf8_outs[i]) for i in range(N_INPUTS)]
            cross_maxd = float(np.max([np.max(np.abs(bf16_outs[i] - bf8_outs[i]))
                                        for i in range(N_INPUTS)]))
            print(f"\n  V_bf16 <-> V_bf8 rounding profile:")
            print(f"    avg cos = {float(np.mean(cross_cos)):.6f}  "
                  f"max|Δ| = {cross_maxd:.4e}")

        # Decision summary
        print("\n" + "=" * 70)
        print("DECISION")
        print("=" * 70)
        if "V_bf16  (all bf16)" in all_results and "V_bf8   (all bf8 [PROD])" in all_results:
            bf16_med = all_results["V_bf16  (all bf16)"][0]
            bf8_med  = all_results["V_bf8   (all bf8 [PROD])"][0]
            speedup  = bf16_med / bf8_med if bf8_med > 0 else float("nan")
            bf16_outs = all_results["V_bf16  (all bf16)"][3]
            bf8_outs  = all_results["V_bf8   (all bf8 [PROD])"][3]
            cos16 = float(np.mean([cosine(oracle_outputs[i], bf16_outs[i]) for i in range(N_INPUTS)]))
            cos8  = float(np.mean([cosine(oracle_outputs[i], bf8_outs[i])  for i in range(N_INPUTS)]))
            delta_cos = cos8 - cos16
            print(f"  bf16 ms/step = {bf16_med:.3f}")
            print(f"  bf8  ms/step = {bf8_med:.3f}")
            print(f"  bf8 speedup  = {speedup:.3f}x")
            print(f"  Δcos (bf8 - bf16, vs fp32 oracle) = {delta_cos:+.6f}")
            # Project to per-tok savings at 64 layers
            per_tok_saving_ms = (bf16_med - bf8_med) * 64
            print(f"  Projected per-tok saving (×64 layers): {per_tok_saving_ms:+.2f} ms/tok")
            if delta_cos > -1e-4 and speedup >= 1.2:
                print("  -> bf8 MLP weights ALREADY shipping; ablation confirms BENEFICIAL.")
            elif delta_cos > -1e-4 and speedup < 1.2:
                print("  -> bf8 MLP weights: quality preserved, latency win minor.")
            else:
                print("  -> bf8 MLP weights: quality regression. Reconsider production default.")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
