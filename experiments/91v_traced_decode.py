#!/usr/bin/env python3
"""
Experiment 91v — C'4: trace capture for Qwen3.6-27B decode step.

Self-contained benchmark. Loads all 64 layers once (~11 min), then:
  1. Runs eager decode for warmup + 20 measured steps -> median ms/tok
  2. Captures the trace of one decode step
  3. Runs traced decode for warmup + 30 measured steps -> median ms/tok
  4. Validates traced logits match eager (cosine >= 0.9999)
  5. Prints the eager-vs-traced perf delta

Trace-friendly design — every per-step host value is written into a
PRE-ALLOCATED device tensor BEFORE execute_trace. The trace itself sees
only device-resident tensors; no Python ints are baked into op arguments.

Pre-allocated buffers (`update_buffers`):
  - embed_buf [1, HIDDEN] fp32     <- embed_np[token_id]
  - cur_pos_buf [1] int32          <- [cur_pos]
  - cos_row_buf [1, ROTARY_DIM] fp32  <- cos_table[cur_pos, :]
  - sin_row_buf [1, ROTARY_DIM] fp32  <- sin_table[cur_pos, :]
  - index_buf [1, N_KV, 1, HEAD_DIM] int32 <- np.full(... cur_pos)

Reference pattern: experiments/85_8b_full_bfp8.py:303-334
Trace-friendly kernel: gated_attn_step_ondevice_traced in 91f.

Run on qb2:
    cd ~/tt-xla && HF_HOME=$HOME/tt-xla/.cache/hf .venv/bin/python \
        experiments/91v_traced_decode.py
"""
import os, sys, json, time, gc
import numpy as np
sys.path.insert(0, os.path.expanduser("~"))
sys.stdout.reconfigure(line_buffering=True)

import torch
import ttnn
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

# Reuse the kernel implementations from 91f.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_91f", os.path.expanduser("~/tt-xla/experiments/91f_qwen36_27b_full_ondevice.py"))
_91f = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_91f)
deltanet_step_ondevice = _91f.deltanet_step_ondevice
gated_attn_step_ondevice = _91f.gated_attn_step_ondevice          # eager (Python cur_pos)
gated_attn_step_ondevice_traced = _91f.gated_attn_step_ondevice_traced  # trace-friendly
mlp_step_ondevice = _91f.mlp_step_ondevice
load_layer_weights_all = _91f.load_layer_weights_all
upload = _91f.upload

MODEL_ID = "Qwen/Qwen3.6-27B"
EPS = 1e-6
MAX_POS = 256
PROMPT = "The capital of France is"
N_EAGER_WARMUP = 3
N_EAGER_MEASURED = 20
N_TRACED_WARMUP = 5
N_TRACED_MEASURED = 30
COSINE_GATE = 0.9999

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)


def load_embed_lm_head_weights():
    idx_path = hf_hub_download(MODEL_ID, "model.safetensors.index.json")
    with open(idx_path) as f:
        weight_map = json.load(f)['weight_map']
    needed = {
        'embed':        "model.language_model.embed_tokens.weight",
        'final_norm':   "model.language_model.norm.weight",
        'lm_head':      "lm_head.weight",
    }
    by_shard = {}
    for key, tname in needed.items():
        if tname in weight_map:
            by_shard.setdefault(weight_map[tname], []).append((key, tname))
    weights = {}
    for shard, items in by_shard.items():
        path = hf_hub_download(MODEL_ID, shard)
        with safe_open(path, framework="pt") as f:
            for key, tname in items:
                t = f.get_tensor(tname).float().numpy()
                if key == 'lm_head':
                    t = t.T
                if key == 'final_norm':
                    t = t + 1.0   # B'9.5: (1.0 + w) RMSNorm
                weights[key] = t.copy()
    return weights


def main():
    print("=" * 72)
    print("Experiment 91v — C'4 trace capture for Qwen3.6-27B decode")
    print("=" * 72)

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
    N_KV = cfg['n_kv_heads']
    HEAD_DIM = cfg['head_dim']
    KEY_DIM = cfg['n_k_heads'] * cfg['k_dim']
    VAL_DIM = cfg['n_v_heads'] * cfg['v_dim']
    CONV_DIM = 2 * KEY_DIM + VAL_DIM
    ROTARY_DIM = int(HEAD_DIM * cfg['partial_rotary_factor'])

    # --- Tokenize ---
    print(f"\n[1/7] Tokenizer + prompt encode...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    prompt_ids = tok.encode(PROMPT)
    print(f"  {len(prompt_ids)} prompt tokens: {prompt_ids}")

    # --- Host weights for embed / lm_head ---
    print(f"\n[2/7] Loading embed + lm_head + final_norm...")
    eweights = load_embed_lm_head_weights()
    embed_np = eweights['embed']
    final_norm_np = eweights['final_norm']
    lm_head_np = eweights['lm_head']

    # --- Device + per-layer weights ---
    print(f"\n[3/7] Opening device + loading all {NUM_LAYERS} layers (~11 min)...")
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
            print(f"    layer {i:2d} loaded ({time.time()-t_load:.1f}s elapsed)")
    print(f"  all {NUM_LAYERS} layers loaded in {time.time()-t_load:.1f}s")

    # --- Recurrent states (deltanet + KV) ---
    n_deltanet = sum(1 for i in range(NUM_LAYERS) if i % 4 != 3)
    n_attn = NUM_LAYERS - n_deltanet

    def fresh_ssm_states():
        return [
            upload(np.zeros((cfg['n_v_heads'], cfg['k_dim'], cfg['v_dim']), dtype=np.float32),
                   device, dtype=ttnn.float32)
            for _ in range(n_deltanet)
        ]

    def fresh_conv_states():
        return [
            upload(np.zeros((CONV_DIM, cfg['conv_kernel']-1), dtype=np.float32),
                   device, dtype=ttnn.float32)
            for _ in range(n_deltanet)
        ]

    def fresh_kv_caches():
        kv_init = np.zeros((1, N_KV, MAX_POS, HEAD_DIM), dtype=np.float32)
        caches = []
        for _ in range(n_attn):
            kv_k = ttnn.from_torch(torch.from_numpy(kv_init), dtype=ttnn.bfloat16,
                                    device=device, layout=ttnn.TILE_LAYOUT)
            kv_v = ttnn.from_torch(torch.from_numpy(kv_init), dtype=ttnn.bfloat16,
                                    device=device, layout=ttnn.TILE_LAYOUT)
            caches.append([kv_k, kv_v])
        return caches

    # --- RoPE precomputed tables (used for both eager and host-side row lookup) ---
    half_rot = ROTARY_DIM // 2
    freqs = 1.0 / (10_000_000.0 ** (np.arange(half_rot).astype(np.float32) / half_rot))
    positions = np.arange(MAX_POS).astype(np.float32)
    all_angles = positions[:, None] * freqs[None, :]
    cos_all = np.concatenate([np.cos(all_angles), np.cos(all_angles)], axis=-1).astype(np.float32)
    sin_all = np.concatenate([np.sin(all_angles), np.sin(all_angles)], axis=-1).astype(np.float32)
    cos_table_tt = upload(cos_all, device, dtype=ttnn.float32)  # eager path slices this
    sin_table_tt = upload(sin_all, device, dtype=ttnn.float32)

    # --- Pre-allocated step buffers (trace-friendly) ---
    embed_buf = upload(np.zeros((1, HIDDEN), dtype=np.float32),
                       device, dtype=ttnn.float32)
    cos_row_buf = upload(np.zeros((1, ROTARY_DIM), dtype=np.float32),
                         device, dtype=ttnn.float32)
    sin_row_buf = upload(np.zeros((1, ROTARY_DIM), dtype=np.float32),
                         device, dtype=ttnn.float32)
    cur_pos_buf = ttnn.from_torch(torch.tensor([0], dtype=torch.int32), device=device)
    index_buf = ttnn.from_torch(
        torch.from_numpy(np.zeros((1, N_KV, 1, HEAD_DIM), dtype=np.int32)),
        dtype=ttnn.int32, device=device, layout=ttnn.TILE_LAYOUT)

    def update_buffers(token_id, cur_pos):
        """Write per-step inputs into PRE-ALLOCATED device tensors. Called BOTH
        in eager mode (before forward) and traced mode (before execute_trace).
        Both paths see the same buffers, so a correctness-equivalent trace
        produces identical numerics."""
        # embed row
        x_np = embed_np[token_id].reshape(1, HIDDEN).astype(np.float32)
        src_embed = ttnn.from_torch(torch.from_numpy(x_np), dtype=ttnn.float32,
                                     layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(src_embed, embed_buf)
        # RoPE row (host lookup from precomputed table — equivalent to the
        # on-device ttnn.slice the eager path uses, but trace-friendly).
        cos_np = cos_all[cur_pos:cur_pos+1, :].astype(np.float32)
        sin_np = sin_all[cur_pos:cur_pos+1, :].astype(np.float32)
        src_cos = ttnn.from_torch(torch.from_numpy(cos_np), dtype=ttnn.float32,
                                    layout=ttnn.TILE_LAYOUT)
        src_sin = ttnn.from_torch(torch.from_numpy(sin_np), dtype=ttnn.float32,
                                    layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(src_cos, cos_row_buf)
        ttnn.copy_host_to_device_tensor(src_sin, sin_row_buf)
        # cur_pos scalar (int32) — SDPA decode reads this
        src_pos = ttnn.from_torch(torch.tensor([cur_pos], dtype=torch.int32))
        ttnn.copy_host_to_device_tensor(src_pos, cur_pos_buf)
        # scatter index [1, N_KV, 1, HEAD_DIM] full of cur_pos
        idx_np = np.full((1, N_KV, 1, HEAD_DIM), cur_pos, dtype=np.int32)
        src_idx = ttnn.from_torch(torch.from_numpy(idx_np), dtype=ttnn.int32,
                                    layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(src_idx, index_buf)

    # --- Forward, EAGER variant. Takes Python cur_pos; slices RoPE on-device
    #     (the existing 91l pattern). Used for prefill and as the correctness
    #     reference.
    def forward_one_token_eager(token_id, cur_pos, ssm_states, conv_states, kv_caches):
        x_np = embed_np[token_id]
        x_tt = upload(x_np.reshape(1, HIDDEN), device, dtype=ttnn.float32)
        cos_tt = ttnn.slice(cos_table_tt, [cur_pos, 0], [cur_pos + 1, ROTARY_DIM])
        sin_tt = ttnn.slice(sin_table_tt, [cur_pos, 0], [cur_pos + 1, ROTARY_DIM])
        cur_pos_tt = ttnn.from_torch(
            torch.tensor([cur_pos], dtype=torch.int32), device=device)
        dn_idx = 0
        attn_idx = 0
        for i in range(NUM_LAYERS):
            layer_type, w_tt = layer_weights[i]
            if layer_type == 'linear_attention':
                x_tt, H_new, c_new = deltanet_step_ondevice(
                    x_tt, w_tt, ssm_states[dn_idx], conv_states[dn_idx], cfg)
                ssm_states[dn_idx] = H_new
                conv_states[dn_idx] = c_new
                dn_idx += 1
            else:
                kv_k, kv_v = kv_caches[attn_idx]
                x_tt, kv_k, kv_v = gated_attn_step_ondevice(
                    x_tt, w_tt, kv_k, kv_v, None, cur_pos_tt, cur_pos,
                    cos_tt, sin_tt, cfg, device)
                kv_caches[attn_idx] = [kv_k, kv_v]
                attn_idx += 1
            x_tt = mlp_step_ondevice(x_tt, w_tt)
        x_tt = ttnn.rms_norm(x_tt, weight=final_norm_tt, epsilon=EPS)
        logits_tt = ttnn.linear(x_tt, lm_head_tt, compute_kernel_config=hifi4)
        return logits_tt

    # --- Forward, TRACE variant. Reads ONLY from pre-allocated device tensors.
    #     Returns the logits tensor (kept alive for ttnn.execute_trace to fill).
    #
    # STATE-ALIASING NOTE (important for understanding what this trace means):
    # Both deltanet_step_ondevice and the scatter inside
    # gated_attn_step_ondevice_traced are FUNCTIONAL ops that return NEW tensors.
    # The trace captures a linear DAG of tensor-producing ops; each replay reads
    # the same input tensors and produces the same output tensor identities.
    # Consequence: across multiple execute_trace replays the SSM state and KV
    # cache CONTENTS do not persist (the original input tensors are unchanged
    # by the functional ops). The traced compute graph is correct for ONE
    # decode step; replaying it back-to-back is FINE FOR TIMING (compute work
    # is identical per replay) but NOT a real generation loop (later tokens
    # are wrong because the cache doesn't accumulate). To get a real
    # generation loop with trace we would need in-place variants
    # (ttnn.experimental.paged_update_cache for KV; ttnn.copy back into
    # ssm_states[] for the SSM recurrence) — a follow-up refactor beyond C'4.
    # The PERF measurement here is still meaningful (compute graph identical;
    # measuring dispatch overhead delta eager vs traced) and the CORRECTNESS
    # gate (cosine on step 0 from identical state) is exactly the apples-to-
    # apples comparison the plan calls for.
    def forward_one_token_traced(ssm_states, conv_states, kv_caches):
        x_tt = embed_buf  # alias; ops produce new tensors so we don't write to it
        dn_idx = 0
        attn_idx = 0
        for i in range(NUM_LAYERS):
            layer_type, w_tt = layer_weights[i]
            if layer_type == 'linear_attention':
                x_tt, H_new, c_new = deltanet_step_ondevice(
                    x_tt, w_tt, ssm_states[dn_idx], conv_states[dn_idx], cfg)
                ssm_states[dn_idx] = H_new
                conv_states[dn_idx] = c_new
                dn_idx += 1
            else:
                kv_k, kv_v = kv_caches[attn_idx]
                x_tt, kv_k, kv_v = gated_attn_step_ondevice_traced(
                    x_tt, w_tt, kv_k, kv_v,
                    cur_pos_buf, cos_row_buf, sin_row_buf, index_buf, cfg)
                kv_caches[attn_idx] = [kv_k, kv_v]
                attn_idx += 1
            x_tt = mlp_step_ondevice(x_tt, w_tt)
        x_tt = ttnn.rms_norm(x_tt, weight=final_norm_tt, epsilon=EPS)
        logits_tt = ttnn.linear(x_tt, lm_head_tt, compute_kernel_config=hifi4)
        return logits_tt

    def logits_to_np(logits_tt):
        return ttnn.to_torch(logits_tt).float().numpy().flatten()[:VOCAB]

    # --- Prefill (eager) — runs once and seeds the state for the decode benchmark.
    print(f"\n[4/7] Prefill ({len(prompt_ids)} tokens, eager)...")
    ssm_states = fresh_ssm_states()
    conv_states = fresh_conv_states()
    kv_caches = fresh_kv_caches()
    t0 = time.time()
    last_logits_tt = None
    for pos, tid in enumerate(prompt_ids):
        last_logits_tt = forward_one_token_eager(tid, pos, ssm_states, conv_states, kv_caches)
    ttnn.synchronize_device(device)
    prefill_logits_np = logits_to_np(last_logits_tt)
    next_id_after_prefill = int(np.argmax(prefill_logits_np))
    prefill_pos = len(prompt_ids)
    print(f"  prefill: {time.time()-t0:.1f}s; first generated token "
          f"{next_id_after_prefill} ({tok.decode([next_id_after_prefill])!r})")

    # Snapshot end-of-prefill state to host. Both the eager benchmark and the
    # traced benchmark restore from this snapshot so they start identically.
    def snapshot_states():
        ssm_np = [ttnn.to_torch(s).float().numpy() for s in ssm_states]
        conv_np = [ttnn.to_torch(s).float().numpy() for s in conv_states]
        kv_np = [(ttnn.to_torch(k).float().numpy(),
                  ttnn.to_torch(v).float().numpy()) for k, v in kv_caches]
        return ssm_np, conv_np, kv_np

    def restore_states(ssm_np, conv_np, kv_np):
        """Build FRESH device tensors from a host snapshot. Used to seed an
        independent benchmark run. Returns new list/tuple containers; callers
        must rebind to the new tensors (`ssm_states = restore_states(...)`)."""
        ss = [upload(s.reshape(cfg['n_v_heads'], cfg['k_dim'], cfg['v_dim']),
                     device, dtype=ttnn.float32) for s in ssm_np]
        cs = [upload(s.reshape(CONV_DIM, cfg['conv_kernel']-1),
                     device, dtype=ttnn.float32) for s in conv_np]
        kvs = []
        for k_np, v_np in kv_np:
            kt = ttnn.from_torch(torch.from_numpy(k_np.reshape(1, N_KV, MAX_POS, HEAD_DIM)),
                                  dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
            vt = ttnn.from_torch(torch.from_numpy(v_np.reshape(1, N_KV, MAX_POS, HEAD_DIM)),
                                  dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
            kvs.append([kt, vt])
        return ss, cs, kvs

    ssm_snap, conv_snap, kv_snap = snapshot_states()

    # =================================================================
    # EAGER decode benchmark
    # =================================================================
    print(f"\n[5/7] Eager decode benchmark "
          f"({N_EAGER_WARMUP} warmup + {N_EAGER_MEASURED} measured)...")

    next_id = next_id_after_prefill
    cur_pos = prefill_pos
    eager_token_log = []  # (next_id, top1_logit_value)
    eager_first_logits = None  # captured at step 0 for cosine check
    eager_times = []

    for step in range(N_EAGER_WARMUP + N_EAGER_MEASURED):
        t0 = time.perf_counter()
        logits_tt = forward_one_token_eager(next_id, cur_pos, ssm_states, conv_states, kv_caches)
        ttnn.synchronize_device(device)
        t1 = time.perf_counter()
        logits = logits_to_np(logits_tt)
        if step == 0:
            eager_first_logits = logits.copy()
        new_next_id = int(np.argmax(logits))
        eager_token_log.append(new_next_id)
        if step >= N_EAGER_WARMUP:
            eager_times.append((t1 - t0) * 1000.0)
        if step < 3 or step == N_EAGER_WARMUP + N_EAGER_MEASURED - 1:
            print(f"    eager step {step:2d}: cur_pos={cur_pos} tok={new_next_id} "
                  f"({tok.decode([new_next_id])!r}) dt={(t1-t0)*1000:.1f} ms")
        next_id = new_next_id
        cur_pos += 1

    eager_median = float(np.median(eager_times))
    eager_p95 = float(np.percentile(eager_times, 95))
    eager_mean = float(np.mean(eager_times))
    print(f"\n  EAGER: median={eager_median:.2f} ms/tok  "
          f"mean={eager_mean:.2f}  p95={eager_p95:.2f}")
    print(f"  EAGER tokens generated: {eager_token_log[:10]}...")

    # =================================================================
    # TRACED decode — capture + benchmark
    # =================================================================
    print(f"\n[6/7] Trace capture + traced decode benchmark "
          f"({N_TRACED_WARMUP} warmup + {N_TRACED_MEASURED} measured)...")

    # Restore states to the SAME starting point as eager (snapshot from
    # immediately post-prefill — same point eager_first_logits was measured at).
    # Note: device-buffer allocations AFTER begin_trace_capture corrupt the
    # captured trace (per the runtime warning "Allocating device buffers is
    # unsafe due to the existence of an active trace"). Strategy:
    #   1. Do ALL fresh allocations (restore_states, warmup) BEFORE capture.
    #   2. update_buffers writes HOST tensors via copy_host_to_device_tensor;
    #      it does NOT allocate device buffers, safe after capture.
    #   3. The capture call itself executes the trace once and fills
    #      logits_ref_tt with the step-0 result — we use THAT as the
    #      correctness comparison (no extra execute_trace + state-reset
    #      dance which v1 tried, and which corrupted the trace).
    ssm_states, conv_states, kv_caches = restore_states(ssm_snap, conv_snap, kv_snap)

    # Warmup pass (untraced) — JITs all ops + primes program cache. Runs ONE
    # full forward through the trace-friendly kernel, advancing state by 1.
    print(f"  warmup pass (untraced, primes program cache)...")
    update_buffers(next_id_after_prefill, prefill_pos)
    _warmup_logits = forward_one_token_traced(ssm_states, conv_states, kv_caches)
    ttnn.synchronize_device(device)
    try:
        device.enable_program_cache()
    except Exception:
        pass
    del _warmup_logits
    gc.collect()

    # Restore from snapshot AGAIN — the warmup mutated state lists (rebinding
    # to new tensor handles). For the trace, we want known state at start.
    # This is the LAST fresh-device-allocation point before trace capture.
    ssm_states, conv_states, kv_caches = restore_states(ssm_snap, conv_snap, kv_snap)
    gc.collect()

    # Capture. Writes inputs into pre-allocated buffers, then captures one
    # forward pass. The capture pass itself executes — logits_ref_tt holds
    # the result of running ONE decode step from post-prefill state. That IS
    # the apples-to-apples comparison to eager_first_logits.
    update_buffers(next_id_after_prefill, prefill_pos)
    print(f"  begin_trace_capture...")
    t_cap = time.time()
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    logits_ref_tt = forward_one_token_traced(ssm_states, conv_states, kv_caches)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    print(f"  trace captured in {time.time()-t_cap:.1f}s; trace_id={tid}")

    # === Correctness gate ===
    # IMPORTANT semantics (v2 run discovery): the begin/end_trace_capture
    # window RECORDS ops but does NOT execute them on the device. The contents
    # of logits_ref_tt are only filled by ttnn.execute_trace. So we run the
    # trace ONCE here — the input buffers were set by the pre-capture
    # update_buffers call, and state is at the snapshot — and read its output.
    # That replay is the apples-to-apples comparison to eager step 0.
    print(f"\n  validating traced output (first execute_trace replay) vs eager step 0...")
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    traced_first_logits = logits_to_np(logits_ref_tt)

    def cosine(a, b):
        a = a.astype(np.float64).flatten()
        b = b.astype(np.float64).flatten()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    cos = cosine(eager_first_logits, traced_first_logits)
    eager_top1 = int(np.argmax(eager_first_logits))
    traced_top1 = int(np.argmax(traced_first_logits))
    print(f"  cosine(traced, eager) = {cos:.6f}")
    print(f"  top-1: eager={eager_top1} ({tok.decode([eager_top1])!r})  "
          f"traced={traced_top1} ({tok.decode([traced_top1])!r})")
    correctness_pass = cos >= COSINE_GATE and eager_top1 == traced_top1

    if correctness_pass:
        print(f"  CORRECTNESS GATE PASS (cosine {cos:.6f} >= {COSINE_GATE})")
    else:
        print(f"\n  ! CORRECTNESS GATE FAILED (cosine {cos:.6f} < {COSINE_GATE} or top1 mismatch)")
        print(f"  Will still run perf benchmark to report observed delta, but the")
        print(f"  traced numerics need fixing before this is production-usable.")

    # Perf benchmark — replay the trace N times. After the validation replay
    # above, state has drifted (the first scatter wrote into the cache at
    # cur_pos=prefill_pos, but later replays' scatter outputs are in
    # transient tensors not connected back). Compute work per replay is
    # identical, which is what we're measuring. Token log is informational.
    # NOTE: we do not call any device-allocating function (upload, fresh
    # from_torch with device=) here — only update_buffers (host-only) and
    # execute_trace. This is required to keep the captured trace valid.
    next_id = next_id_after_prefill
    cur_pos = prefill_pos

    traced_token_log = []
    traced_times = []   # full-step time (update_buffers + execute_trace + readback)
    traced_exec_times = []  # execute_trace alone

    for step in range(N_TRACED_WARMUP + N_TRACED_MEASURED):
        t0 = time.perf_counter()
        update_buffers(next_id, cur_pos)
        t1 = time.perf_counter()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        t2 = time.perf_counter()
        logits = logits_to_np(logits_ref_tt)
        new_next_id = int(np.argmax(logits))
        t3 = time.perf_counter()
        traced_token_log.append(new_next_id)
        if step >= N_TRACED_WARMUP:
            traced_times.append((t3 - t0) * 1000.0)
            traced_exec_times.append((t2 - t1) * 1000.0)
        if step < 3 or step == N_TRACED_WARMUP + N_TRACED_MEASURED - 1:
            print(f"    traced step {step:2d}: cur_pos={cur_pos} tok={new_next_id} "
                  f"({tok.decode([new_next_id])!r}) "
                  f"dt_total={(t3-t0)*1000:.2f} dt_exec={(t2-t1)*1000:.2f} ms")
        next_id = new_next_id
        cur_pos += 1

    traced_median = float(np.median(traced_times))
    traced_p95 = float(np.percentile(traced_times, 95))
    traced_mean = float(np.mean(traced_times))
    traced_exec_median = float(np.median(traced_exec_times))
    print(f"\n  TRACED: median={traced_median:.2f} ms/tok  "
          f"mean={traced_mean:.2f}  p95={traced_p95:.2f}  "
          f"(execute_trace alone: median={traced_exec_median:.2f})")
    print(f"  TRACED tokens generated: {traced_token_log[:10]}...")

    ttnn.release_trace(device, tid)

    # =================================================================
    # Report
    # =================================================================
    print(f"\n[7/7] " + "=" * 64)
    print(f"  C'4 RESULTS")
    print("=" * 72)
    print(f"  Eager  median ms/tok: {eager_median:.2f}")
    print(f"  Traced median ms/tok: {traced_median:.2f}")
    delta = traced_median - eager_median
    speedup = eager_median / traced_median if traced_median > 0 else float('nan')
    print(f"  Delta:                {delta:+.2f} ms/tok  "
          f"({(1.0 - traced_median/eager_median)*100:+.1f}%)")
    print(f"  Speedup:              {speedup:.2f}x")
    print(f"  Correctness:          cosine={cos:.6f}  "
          f"top1_match={'YES' if eager_top1==traced_top1 else 'NO'}")
    # Token agreement on the first few steps
    common = min(len(eager_token_log), len(traced_token_log), 5)
    eager_head = eager_token_log[:common]
    traced_head = traced_token_log[:common]
    print(f"  First {common} tokens (eager):  {eager_head}")
    print(f"  First {common} tokens (traced): {traced_head}")
    print(f"  Match:                {'YES' if eager_head == traced_head else 'NO'}")
    print("=" * 72)

    # Dump structured results for the writeup.
    out = {
        'eager_ms_median': eager_median,
        'eager_ms_mean': eager_mean,
        'eager_ms_p95': eager_p95,
        'traced_ms_median': traced_median,
        'traced_ms_mean': traced_mean,
        'traced_ms_p95': traced_p95,
        'traced_exec_ms_median': traced_exec_median,
        'delta_ms': delta,
        'speedup': speedup,
        'cosine': cos,
        'eager_top1': eager_top1,
        'traced_top1': traced_top1,
        'eager_tokens_head': eager_head,
        'traced_tokens_head': traced_head,
        'eager_token_log': eager_token_log,
        'traced_token_log': traced_token_log,
        'n_eager_measured': N_EAGER_MEASURED,
        'n_traced_measured': N_TRACED_MEASURED,
        'prompt': PROMPT,
        'prompt_ids': prompt_ids,
        'cosine_gate': COSINE_GATE,
        'correctness_pass': correctness_pass,
    }
    out_path = os.path.expanduser("~/tt-xla/.cache/c4_traced_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  results -> {out_path}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
