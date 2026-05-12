#!/usr/bin/env python3
"""
Performance baseline harness for Qwen3.6-27B (Branch III).

Used to measure where time goes BEFORE applying any perf phase (C'1, C'2, ...).
Re-run after each phase to quantify the improvement.

# Why this exists (the async-timing trap)

Host-side timing of GPU/accelerator ops is BROKEN when you don't synchronize:

    t0 = time.time()
    result = ttnn.linear(x, weight, ...)   # submits to queue, returns immediately
    t1 = time.time()
    # t1 - t0 measures DISPATCH time, NOT kernel execution time!

For slow ops (matmul), dispatch might be 50 µs but kernel execute might be
50 ms — host reports the 50 µs and you have no idea the op cost 1000× more.
The measurement is wrong by orders of magnitude and worse, inconsistent.

The fix: synchronize BEFORE and AFTER:

    ttnn.synchronize_device(device)   # drain queue
    t0 = time.time()
    result = ttnn.linear(...)
    ttnn.synchronize_device(device)   # block until this op done
    t1 = time.time()
    # t1 - t0 = full submit + execute + final-sync. END-TO-END.

This harness wraps every measurement region with that sync-before/sync-after
discipline. It also runs warmup iterations (JIT compile, L1 cold) and reports
median across repeats (more robust than mean for noisy timing).

# What this measures and what it doesn't

MEASURES (correctly):
  - End-to-end wall time for a region (submit + execute + final sync)
  - Median + min + max across repeats so outliers are visible

DOES NOT MEASURE (would need Tracy device profiler):
  - Pure kernel execution time (separated from submit/sync overhead)
  - Per-Tensix-core utilization
  - DRAM bandwidth saturation
  - PCIe queue depth

For comparing 'before-C'1 vs after-C'1' on the full decode step, end-to-end
is correct and sufficient. For zooming inside a single matmul we'd need Tracy.

# Output

  - Markdown table to stdout (human-readable)
  - JSON to ~/tt-xla/.cache/perf_baseline_<phase>_<timestamp>.json

Use the JSON to diff across phases programmatically.

Run on qb2:
    cd ~/tt-xla && HF_HOME=$HOME/tt-xla/.cache/hf .venv/bin/python \\
        experiments/utils/perf_baseline.py --phase C0
"""
import os, sys, json, time, gc, argparse, statistics
from contextlib import contextmanager
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import torch
import ttnn
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

# Reuse 91f/91l production kernels (all 7 bug fixes baked in)
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_91f", os.path.expanduser("~/tt-xla/experiments/91f_qwen36_27b_full_ondevice.py"))
_91f = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_91f)
deltanet_step_ondevice = _91f.deltanet_step_ondevice
gated_attn_step_ondevice = _91f.gated_attn_step_ondevice
mlp_step_ondevice = _91f.mlp_step_ondevice
load_layer_weights_all = _91f.load_layer_weights_all
upload = _91f.upload

_spec2 = importlib.util.spec_from_file_location(
    "_91l", os.path.expanduser("~/tt-xla/experiments/91l_fp32_residual_generate.py"))
_91l = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_91l)
load_embed_lm_head_weights = _91l.load_embed_lm_head_weights

MODEL_ID = "Qwen/Qwen3.6-27B"
EPS = 1e-6
MAX_POS = 256
PROMPT = "The capital of France is"

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)


@contextmanager
def timed(device, label, results_dict, repeats=10, warmup=3):
    """Time a code region with proper sync-before-and-after discipline.

    Usage:
        with timed(device, "decode_step", results) as t:
            forward_token(...)
        # `t.elapsed_ms` is the median across `repeats` runs after `warmup` discarded runs

    Notes:
      - Caller is responsible for the loop. The context just wraps ONE invocation.
      - For multi-repeat timing, see `time_function` below.
    """
    ttnn.synchronize_device(device)
    t0 = time.time()
    class _T: pass
    timing = _T()
    yield timing
    ttnn.synchronize_device(device)
    timing.elapsed_ms = (time.time() - t0) * 1000
    results_dict.setdefault(label, []).append(timing.elapsed_ms)


def time_function(device, label, fn, results_dict, repeats=10, warmup=3):
    """Run `fn()` `warmup + repeats` times. Discard warmups. Record per-call
    elapsed_ms with sync-bounded measurement. fn() may return anything; we
    ignore the value (you're measuring its side-effects).
    """
    # Warmup (not measured)
    for _ in range(warmup):
        ttnn.synchronize_device(device)
        fn()
    ttnn.synchronize_device(device)
    # Timed
    times = []
    for _ in range(repeats):
        ttnn.synchronize_device(device)
        t0 = time.time()
        fn()
        ttnn.synchronize_device(device)
        times.append((time.time() - t0) * 1000)
    results_dict[label] = times
    return times


def stats(times_ms):
    """Return median / min / max / std for a list of times."""
    if not times_ms:
        return {"n": 0}
    return {
        "n": len(times_ms),
        "median_ms": statistics.median(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "stdev_ms": (statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", default="C0",
                   help="Phase label for output filename (C0=baseline, C1=after paged_update_cache, etc.)")
    p.add_argument("--prefill-repeats", type=int, default=3, help="Prefill runs (with full state reset)")
    p.add_argument("--decode-repeats", type=int, default=20,
                   help="Decode-step runs (after one prefill)")
    p.add_argument("--decode-warmup", type=int, default=5)
    p.add_argument("--per-layer-repeats", type=int, default=10,
                   help="Repeats for per-layer-type sampling")
    args = p.parse_args()

    print("=" * 64)
    print(f"Perf baseline harness — phase={args.phase}")
    print("=" * 64)

    # ----------------------------------------
    # Config + model load (same as 91l/demo)
    # ----------------------------------------
    cfg_path = hf_hub_download(MODEL_ID, "config.json")
    with open(cfg_path) as f:
        text_cfg = json.load(f)['text_config']
    cfg = {
        'hidden':      text_cfg['hidden_size'],
        'n_k_heads':   text_cfg['linear_num_key_heads'],
        'n_v_heads':   text_cfg['linear_num_value_heads'],
        'k_dim':       text_cfg['linear_key_head_dim'],
        'v_dim':       text_cfg['linear_value_head_dim'],
        'conv_kernel': text_cfg['linear_conv_kernel_dim'],
        'n_q_heads':   text_cfg['num_attention_heads'],
        'n_kv_heads':  text_cfg['num_key_value_heads'],
        'head_dim':    text_cfg['head_dim'],
        'partial_rotary_factor': text_cfg['partial_rotary_factor'],
    }
    NUM_LAYERS = text_cfg['num_hidden_layers']
    HIDDEN = cfg['hidden']
    VOCAB = text_cfg['vocab_size']
    KEY_DIM = cfg['n_k_heads'] * cfg['k_dim']
    VAL_DIM = cfg['n_v_heads'] * cfg['v_dim']
    CONV_DIM = 2 * KEY_DIM + VAL_DIM

    print(f"\n[1/4] Loading tokenizer + embed + lm_head…")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    eweights = load_embed_lm_head_weights()
    embed_np = eweights['embed']
    final_norm_np = eweights['final_norm']
    lm_head_np = eweights['lm_head']

    print(f"\n[2/4] Opening device + loading all {NUM_LAYERS} layers — ~10 min…")
    device = ttnn.open_device(device_id=0)
    final_norm_tt = upload(final_norm_np, device, dtype=ttnn.bfloat16)
    lm_head_tt = upload(lm_head_np, device, dtype=ttnn.bfloat8_b)

    t_load = time.time()
    layer_weights = []
    for i in range(NUM_LAYERS):
        layer_type = 'linear_attention' if i % 4 != 3 else 'full_attention'
        w_np = load_layer_weights_all(i, layer_type)
        w_tt = {}
        for k, arr in w_np.items():
            if k == 'conv1d_weight' and arr.ndim == 3:
                arr = arr.squeeze(1)
            if 'proj' in k or k == 'conv1d_weight':
                dt = ttnn.bfloat8_b
            elif k in ('A_log', 'dt_bias'):
                dt = ttnn.float32
            else:
                dt = ttnn.bfloat16
            w_tt[k] = upload(arr, device, dtype=dt)
        layer_weights.append((layer_type, w_tt))
        del w_np
        gc.collect()
        if i % 16 == 0 or i == NUM_LAYERS - 1:
            print(f"    layer {i:2d}  ({time.time()-t_load:.0f}s)")
    print(f"  ✓ all {NUM_LAYERS} layers loaded in {time.time()-t_load:.0f}s")

    # ----------------------------------------
    # State helpers
    # ----------------------------------------
    rotary_dim = int(cfg['head_dim'] * cfg['partial_rotary_factor'])
    half_rot = rotary_dim // 2
    freqs = 1.0 / (10_000_000.0 ** (np.arange(half_rot).astype(np.float32) / half_rot))

    def rope_for_pos(pos):
        angles = pos * freqs
        cos_np = np.concatenate([np.cos(angles), np.cos(angles)]).astype(np.float32)
        sin_np = np.concatenate([np.sin(angles), np.sin(angles)]).astype(np.float32)
        return (upload(cos_np, device, dtype=ttnn.float32),
                upload(sin_np, device, dtype=ttnn.float32))

    def fresh_state():
        n_dn = sum(1 for i in range(NUM_LAYERS) if i % 4 != 3)
        n_attn = NUM_LAYERS - n_dn
        ssm = [upload(np.zeros((cfg['n_v_heads'], cfg['k_dim'], cfg['v_dim']), dtype=np.float32),
                      device, dtype=ttnn.float32) for _ in range(n_dn)]
        cvs = [upload(np.zeros((CONV_DIM, cfg['conv_kernel']-1), dtype=np.float32),
                      device, dtype=ttnn.float32) for _ in range(n_dn)]
        kvc = []
        kv_init = np.zeros((1, cfg['n_kv_heads'], MAX_POS, cfg['head_dim']), dtype=np.float32)
        for _ in range(n_attn):
            kv_k = ttnn.from_torch(torch.from_numpy(kv_init), dtype=ttnn.bfloat16,
                                    device=device, layout=ttnn.TILE_LAYOUT)
            kv_v = ttnn.from_torch(torch.from_numpy(kv_init), dtype=ttnn.bfloat16,
                                    device=device, layout=ttnn.TILE_LAYOUT)
            kvc.append([kv_k, kv_v])
        return ssm, cvs, kvc

    def forward_token(token_id, cur_pos, ssm, cvs, kvc):
        """Full single-token forward through all 64 layers. Returns logits."""
        x_np = embed_np[token_id]
        x_tt = upload(x_np.reshape(1, HIDDEN), device, dtype=ttnn.float32)
        cos_tt, sin_tt = rope_for_pos(cur_pos)
        cur_pos_tt = ttnn.from_torch(torch.tensor([cur_pos], dtype=torch.int32), device=device)
        dn_idx = 0
        attn_idx = 0
        for i in range(NUM_LAYERS):
            layer_type, w_tt = layer_weights[i]
            if layer_type == 'linear_attention':
                x_tt, ssm[dn_idx], cvs[dn_idx] = deltanet_step_ondevice(
                    x_tt, w_tt, ssm[dn_idx], cvs[dn_idx], cfg)
                dn_idx += 1
            else:
                kv_k, kv_v = kvc[attn_idx]
                x_tt, kv_k, kv_v = gated_attn_step_ondevice(
                    x_tt, w_tt, kv_k, kv_v, None, cur_pos_tt, cur_pos,
                    cos_tt, sin_tt, cfg, device)
                kvc[attn_idx] = [kv_k, kv_v]
                attn_idx += 1
            x_tt = mlp_step_ondevice(x_tt, w_tt)
        x_tt = ttnn.rms_norm(x_tt, weight=final_norm_tt, epsilon=EPS)
        logits_tt = ttnn.linear(x_tt, lm_head_tt, compute_kernel_config=hifi4)
        return logits_tt

    # ----------------------------------------
    # Benchmarks
    # ----------------------------------------
    results = {}      # label → list of elapsed_ms
    prompt_ids = tok.encode(PROMPT)

    print(f"\n[3/4] Running measurements…")

    # ----- (a) Full prefill (state reset per run) -----
    print(f"  • prefill ({len(prompt_ids)} tokens, full state reset per run, "
          f"{args.prefill_repeats} repeats)…")
    def do_prefill():
        ssm, cvs, kvc = fresh_state()
        for pos, tid in enumerate(prompt_ids):
            _ = forward_token(tid, pos, ssm, cvs, kvc)
    time_function(device, "prefill_5_tokens", do_prefill, results,
                  repeats=args.prefill_repeats, warmup=1)

    # ----- (b) Single decode step (after one prefill, fixed cur_pos) -----
    # We prefill once, then measure repeated single-step decode calls. State
    # is reused across repeats — the cache continues to grow, which slightly
    # changes the SDPA cost per token. To control this, we reset state each
    # repeat. Otherwise pos N would be measuring a slightly different op than
    # pos N+1.
    print(f"  • decode_step_single (one prefill, fresh state each repeat, "
          f"{args.decode_repeats} repeats)…")
    def do_single_decode():
        ssm, cvs, kvc = fresh_state()
        for pos, tid in enumerate(prompt_ids):
            _ = forward_token(tid, pos, ssm, cvs, kvc)
        # Now measure one decode step at position len(prompt_ids)
        _ = forward_token(prompt_ids[-1], len(prompt_ids) - 1, ssm, cvs, kvc)
    # Each call here is one prefill (5x forward) + one decode step. We'll
    # subtract off prefill timing afterwards to isolate the decode step.
    time_function(device, "prefill_plus_one_decode", do_single_decode, results,
                  repeats=args.decode_repeats, warmup=args.decode_warmup)

    # ----- (c) Per-layer-type sample: time ONE DeltaNet step in isolation -----
    print(f"  • single deltanet_step (layer 0 weights, {args.per_layer_repeats} repeats)…")
    ssm_one = upload(np.zeros((cfg['n_v_heads'], cfg['k_dim'], cfg['v_dim']), dtype=np.float32),
                     device, dtype=ttnn.float32)
    cvs_one = upload(np.zeros((CONV_DIM, cfg['conv_kernel']-1), dtype=np.float32),
                     device, dtype=ttnn.float32)
    x_one = upload(embed_np[prompt_ids[0]].reshape(1, HIDDEN), device, dtype=ttnn.float32)
    w_dn = layer_weights[0][1]
    def do_one_deltanet():
        _ = deltanet_step_ondevice(x_one, w_dn, ssm_one, cvs_one, cfg)
    time_function(device, "single_deltanet_step", do_one_deltanet, results,
                  repeats=args.per_layer_repeats, warmup=3)

    # ----- (d) Per-layer-type sample: time ONE full-attention step -----
    print(f"  • single gated_attn_step (layer 3 weights, {args.per_layer_repeats} repeats)…")
    kv_init = np.zeros((1, cfg['n_kv_heads'], MAX_POS, cfg['head_dim']), dtype=np.float32)
    kv_k_one = ttnn.from_torch(torch.from_numpy(kv_init), dtype=ttnn.bfloat16,
                                device=device, layout=ttnn.TILE_LAYOUT)
    kv_v_one = ttnn.from_torch(torch.from_numpy(kv_init), dtype=ttnn.bfloat16,
                                device=device, layout=ttnn.TILE_LAYOUT)
    cos_one, sin_one = rope_for_pos(0)
    cur_pos_one = ttnn.from_torch(torch.tensor([0], dtype=torch.int32), device=device)
    w_fa = layer_weights[3][1]
    def do_one_gated_attn():
        _ = gated_attn_step_ondevice(x_one, w_fa, kv_k_one, kv_v_one, None,
                                      cur_pos_one, 0, cos_one, sin_one, cfg, device)
    time_function(device, "single_gated_attn_step", do_one_gated_attn, results,
                  repeats=args.per_layer_repeats, warmup=3)

    # ----- (e) Per-layer-type sample: time ONE MLP step -----
    print(f"  • single mlp_step (layer 0 weights, {args.per_layer_repeats} repeats)…")
    def do_one_mlp():
        _ = mlp_step_ondevice(x_one, w_dn)
    time_function(device, "single_mlp_step", do_one_mlp, results,
                  repeats=args.per_layer_repeats, warmup=3)

    # ----- (f) lm_head matmul cost -----
    print(f"  • lm_head (rms_norm + linear, {args.per_layer_repeats} repeats)…")
    def do_lm_head():
        h = ttnn.rms_norm(x_one, weight=final_norm_tt, epsilon=EPS)
        _ = ttnn.linear(h, lm_head_tt, compute_kernel_config=hifi4)
    time_function(device, "lm_head", do_lm_head, results,
                  repeats=args.per_layer_repeats, warmup=3)

    # ----------------------------------------
    # Derive
    # ----------------------------------------
    # decode_step_one = (prefill_plus_one_decode) - (prefill_5_tokens) ≈ single decode token
    prefill_med = statistics.median(results.get("prefill_5_tokens", [0]))
    bundle_med = statistics.median(results.get("prefill_plus_one_decode", [0]))
    one_decode_ms = bundle_med - prefill_med
    results["decode_step_derived"] = [one_decode_ms]   # list of 1 so .stats() shape matches

    # ----------------------------------------
    # Report
    # ----------------------------------------
    print(f"\n[4/4] Results — phase={args.phase}")
    print("=" * 80)
    print(f"{'metric':>32s}  {'n':>4s}  {'median':>10s}  {'min':>10s}  {'max':>10s}  {'stdev':>10s}")
    print("-" * 80)
    for label in [
        "prefill_5_tokens", "prefill_plus_one_decode", "decode_step_derived",
        "single_deltanet_step", "single_gated_attn_step",
        "single_mlp_step", "lm_head",
    ]:
        s = stats(results.get(label, []))
        if not s.get("n"):
            continue
        print(f"{label:>32s}  {s['n']:>4d}  "
              f"{s['median_ms']:>9.2f}ms  {s['min_ms']:>9.2f}ms  "
              f"{s['max_ms']:>9.2f}ms  {s['stdev_ms']:>9.2f}ms")

    # Layer-type compounding sanity check: does 48*deltanet + 16*(gated_attn) + 64*mlp + lm_head
    # roughly match the derived one-decode-step time?
    n_dn = sum(1 for i in range(NUM_LAYERS) if i % 4 != 3)
    n_attn = NUM_LAYERS - n_dn
    estimate = (n_dn * statistics.median(results["single_deltanet_step"])
                + n_attn * statistics.median(results["single_gated_attn_step"])
                + NUM_LAYERS * statistics.median(results["single_mlp_step"])
                + statistics.median(results["lm_head"]))
    print()
    print(f"Layer-type compounding estimate "
          f"({n_dn}×deltanet + {n_attn}×gated_attn + {NUM_LAYERS}×mlp + lm_head)")
    print(f"  = {estimate:8.2f} ms  vs  measured decode_step = {one_decode_ms:8.2f} ms")
    print(f"  diff: {one_decode_ms - estimate:+.2f} ms "
          f"({100 * (one_decode_ms - estimate) / max(one_decode_ms, 1):+.1f}% — host/dispatch/sync overhead if positive)")

    # ----------------------------------------
    # Persist
    # ----------------------------------------
    out_dir = os.path.expanduser("~/tt-xla/.cache")
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(out_dir, f"perf_baseline_{args.phase}_{ts}.json")
    persisted = {
        "phase": args.phase,
        "timestamp": ts,
        "model": MODEL_ID,
        "prompt": PROMPT,
        "raw_times_ms": results,
        "stats": {k: stats(v) for k, v in results.items()},
        "derived": {
            "decode_step_ms": one_decode_ms,
            "compounding_estimate_ms": estimate,
            "host_overhead_ms": one_decode_ms - estimate,
        },
    }
    with open(out_path, "w") as f:
        json.dump(persisted, f, indent=2)
    print(f"\nSaved → {out_path}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
