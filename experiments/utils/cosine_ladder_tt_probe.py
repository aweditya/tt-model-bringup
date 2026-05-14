#!/usr/bin/env python3
"""
Cosine-ladder TT probe — teacher-force the HF reference sequence through
our paged TT forward on qb1 device 3, capture per-position logits, and
compute cosine + top-1 agreement at each step.

Requires that cosine_ladder_hf_ref.py has already been run and produced
~/tt-xla/.cache/cosine_ladder_hf_ref.npz.

This script does NOT touch the running persistent server. It opens
device 3 directly and replicates the server's bootstrap:
  - Same MODEL_ID
  - Same dtype policy (bf8 projections, bf16 norms, fp32 small scalars,
    bf8 lm_head)
  - Same paged forward (gated_attn_step_ondevice_paged + paged SDPA)
  - Same partial RoPE V2 (rotate-only) via the 91f kernels

The forward path mirrors server.handle_generate_paged. The only
difference: we feed the teacher-forced token sequence INSTEAD of
argmaxing per step.

Output: cosine ladder + per-position top-1 match, printed + saved to
~/tt-xla/.cache/cosine_ladder_tt_results.json

Run on qb1:
    cd ~/tt-xla && .venv/bin/python -u \
        experiments/utils/cosine_ladder_tt_probe.py \
        --device-id 3 --positions 1,25,50,100
"""
import os, sys, json, time, gc, argparse, importlib.util
import numpy as np
import torch
import ttnn
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

MODEL_ID = "Qwen/Qwen3.6-27B"
HF_REF_PATH = os.path.expanduser("~/tt-xla/.cache/cosine_ladder_hf_ref.npz")
OUT_PATH = os.path.expanduser("~/tt-xla/.cache/cosine_ladder_tt_results.json")

# Load the 91f kernel module (same path the server uses)
_91F_PATH = os.path.expanduser(
    "~/tt-xla/experiments/91f_qwen36_27b_full_ondevice.py")
_91L_PATH = os.path.expanduser(
    "~/tt-xla/experiments/91l_fp32_residual_generate.py")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return num / den


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device-id", type=int, default=3,
                   help="Tenstorrent device id (qb1 device 3 by convention).")
    p.add_argument("--max-pos", type=int, default=512,
                   help="Paged KV cache size. Must be ≥ prompt_len + max_tokens.")
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--positions", default="1,10,25,50,75,100,150,200,300,400",
                   help="Comma-separated 1-indexed positions in the generated "
                        "sequence to report cosines for (subset of 1..M).")
    p.add_argument("--ref", default=HF_REF_PATH)
    p.add_argument("--out", default=OUT_PATH)
    args = p.parse_args()

    print("=" * 64, flush=True)
    print(f"Cosine-ladder TT probe   device={args.device_id}", flush=True)
    print(f"  ref:        {args.ref}", flush=True)
    print(f"  max_pos:    {args.max_pos}  block_size: {args.block_size}", flush=True)
    print("=" * 64, flush=True)

    # ----- Load HF reference -----
    if not os.path.exists(args.ref):
        print(f"ERROR: HF reference not found at {args.ref}", flush=True)
        print(f"Run experiments/utils/cosine_ladder_hf_ref.py first.", flush=True)
        sys.exit(2)
    ref = np.load(args.ref, allow_pickle=True)
    prompt_ids = ref["prompt_ids"].astype(int).tolist()
    generated_ids = ref["generated_ids"].astype(int).tolist()
    logits_ref = ref["logits_at_step"]  # [M, V]
    M = len(generated_ids)
    P = len(prompt_ids)
    VOCAB = logits_ref.shape[1]
    ref_dtype = str(ref["dtype"][0])
    prompt = str(ref["prompt"][0])
    print(f"HF ref loaded:  prompt='{prompt}' ({P} ids)  M={M}  vocab={VOCAB}  "
          f"dtype={ref_dtype}", flush=True)
    print(f"  generated_ids[:5] = {generated_ids[:5]}", flush=True)
    print(f"  generated_ids[-5:] = {generated_ids[-5:]}", flush=True)

    # ----- Modules + cfg -----
    print(f"\nLoading kernel modules…", flush=True)
    _91f = load_module(_91F_PATH, "_91f_probe")
    _91l = load_module(_91L_PATH, "_91l_probe")
    upload = _91f.upload
    load_layer_weights_all = _91f.load_layer_weights_all
    load_embed_lm_head_weights = _91l.load_embed_lm_head_weights

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
    NUM_LAYERS = text_cfg["num_hidden_layers"]
    HIDDEN = cfg["hidden"]
    N_KV = cfg["n_kv_heads"]
    HEAD_DIM = cfg["head_dim"]
    ROTARY_DIM = int(HEAD_DIM * cfg["partial_rotary_factor"])
    KEY_DIM = cfg["n_k_heads"] * cfg["k_dim"]
    VAL_DIM = cfg["n_v_heads"] * cfg["v_dim"]
    CONV_DIM = 2 * KEY_DIM + VAL_DIM
    assert args.max_pos >= P + M, (
        f"max_pos {args.max_pos} < prompt_len {P} + max_tokens {M}; "
        f"increase --max-pos")
    assert args.max_pos % args.block_size == 0, (
        f"max_pos {args.max_pos} must be multiple of block_size {args.block_size}")
    print(f"  cfg: NUM_LAYERS={NUM_LAYERS} hidden={HIDDEN} n_kv={N_KV} "
          f"head_dim={HEAD_DIM} rotary_dim={ROTARY_DIM}", flush=True)

    # ----- Embed + lm_head -----
    print(f"\nLoading embed + lm_head + final_norm…", flush=True)
    eweights = load_embed_lm_head_weights()
    embed_np = eweights["embed"]
    final_norm_np = eweights["final_norm"]
    lm_head_np = eweights["lm_head"]

    # ----- Open device + upload weights -----
    print(f"\nOpening device {args.device_id}…", flush=True)
    device = ttnn.open_device(device_id=args.device_id)

    final_norm_tt = upload(final_norm_np, device, dtype=ttnn.bfloat16)
    lm_head_tt = upload(lm_head_np, device, dtype=ttnn.bfloat8_b)

    print(f"\nUploading {NUM_LAYERS} layer weights…", flush=True)
    t0 = time.time()
    layer_weights = []
    for i in range(NUM_LAYERS):
        layer_type = "linear_attention" if i % 4 != 3 else "full_attention"
        w_np = load_layer_weights_all(i, layer_type)
        w_tt = {}
        for k, arr in w_np.items():
            if k == "conv1d_weight" and arr.ndim == 3:
                arr = arr.squeeze(1)
            if "proj" in k or k == "conv1d_weight":
                dt = ttnn.bfloat8_b
            elif k in ("A_log", "dt_bias"):
                dt = ttnn.float32
            else:
                dt = ttnn.bfloat16
            w_tt[k] = upload(arr, device, dtype=dt)
        layer_weights.append((layer_type, w_tt))
        del w_np
        gc.collect()
        if i % 16 == 0 or i == NUM_LAYERS - 1:
            print(f"  layer {i:2d}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"All layers uploaded in {time.time()-t0:.0f}s", flush=True)

    # ----- RoPE tables (covering max_pos) -----
    print(f"\nBuilding RoPE tables for {args.max_pos} positions…", flush=True)
    half_rot = ROTARY_DIM // 2
    freqs = 1.0 / (10_000_000.0 ** (np.arange(half_rot).astype(np.float32) / half_rot))
    positions = np.arange(args.max_pos).astype(np.float32)
    all_angles = positions[:, None] * freqs[None, :]
    cos_all = np.concatenate([np.cos(all_angles), np.cos(all_angles)], axis=-1).astype(np.float32)
    sin_all = np.concatenate([np.sin(all_angles), np.sin(all_angles)], axis=-1).astype(np.float32)
    pad_size = HEAD_DIM - ROTARY_DIM
    cos_ext_pad = np.ones((args.max_pos, pad_size), dtype=np.float32)
    sin_ext_pad = np.zeros((args.max_pos, pad_size), dtype=np.float32)
    cos_ext_all = np.concatenate([cos_all, cos_ext_pad], axis=-1).astype(np.float32)
    sin_ext_all = np.concatenate([sin_all, sin_ext_pad], axis=-1).astype(np.float32)
    cos_ext_table_tt = upload(cos_ext_all, device, dtype=ttnn.float32)
    sin_ext_table_tt = upload(sin_ext_all, device, dtype=ttnn.float32)

    # ----- Fresh paged state -----
    max_num_blocks = args.max_pos // args.block_size
    page_table_np = np.arange(max_num_blocks, dtype=np.int32).reshape(1, max_num_blocks)
    page_table_tt = ttnn.from_torch(torch.from_numpy(page_table_np),
                                     dtype=ttnn.int32, device=device,
                                     layout=ttnn.ROW_MAJOR_LAYOUT)

    n_dn = sum(1 for i in range(NUM_LAYERS) if i % 4 != 3)
    n_attn = NUM_LAYERS - n_dn
    print(f"  n_dn={n_dn}  n_attn={n_attn}", flush=True)

    ssm = [upload(np.zeros((cfg["n_v_heads"], cfg["k_dim"], cfg["v_dim"]),
                            dtype=np.float32), device, dtype=ttnn.float32)
           for _ in range(n_dn)]
    cvs = [upload(np.zeros((CONV_DIM, cfg["conv_kernel"] - 1), dtype=np.float32),
                    device, dtype=ttnn.float32) for _ in range(n_dn)]
    paged_kv_zero = np.zeros((max_num_blocks, N_KV, args.block_size, HEAD_DIM),
                              dtype=np.float32)
    kvc = []
    for _ in range(n_attn):
        kv_k = ttnn.from_torch(torch.from_numpy(paged_kv_zero), dtype=ttnn.bfloat16,
                                device=device, layout=ttnn.TILE_LAYOUT,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
        kv_v = ttnn.from_torch(torch.from_numpy(paged_kv_zero), dtype=ttnn.bfloat16,
                                device=device, layout=ttnn.TILE_LAYOUT,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
        kvc.append([kv_k, kv_v])

    # ----- forward_token (mirrors handle_generate_paged.forward_token) -----
    def forward_token(token_id, cur_pos):
        x_np = embed_np[token_id]
        x_tt = upload(x_np.reshape(1, HIDDEN), device, dtype=ttnn.float32)
        cos_tt = ttnn.slice(cos_ext_table_tt, [cur_pos, 0], [cur_pos + 1, ROTARY_DIM])
        sin_tt = ttnn.slice(sin_ext_table_tt, [cur_pos, 0], [cur_pos + 1, ROTARY_DIM])
        cur_pos_tt = ttnn.from_torch(torch.tensor([cur_pos], dtype=torch.int32),
                                       device=device,
                                       layout=ttnn.ROW_MAJOR_LAYOUT)
        dn_idx = 0
        attn_idx = 0
        for i in range(NUM_LAYERS):
            layer_type, w_tt = layer_weights[i]
            if layer_type == "linear_attention":
                x_tt, ssm[dn_idx], cvs[dn_idx] = _91f.deltanet_step_ondevice(
                    x_tt, w_tt, ssm[dn_idx], cvs[dn_idx], cfg)
                dn_idx += 1
            else:
                kv_k, kv_v = kvc[attn_idx]
                x_tt = _91f.gated_attn_step_ondevice_paged(
                    x_tt, w_tt, kv_k, kv_v, page_table_tt, cur_pos_tt,
                    cos_tt, sin_tt, cfg)
                attn_idx += 1
            x_tt = _91f.mlp_step_ondevice(x_tt, w_tt)
        x_tt = ttnn.rms_norm(x_tt, weight=final_norm_tt, epsilon=1e-6)
        logits_tt = ttnn.linear(x_tt, lm_head_tt, compute_kernel_config=_91f.hifi4)
        return logits_tt

    # ----- Drive teacher-forced sequence -----
    # Step model: for the cosine ladder we need our logits at the SAME
    # positions HF reports.
    #
    # HF reference convention (cosine_ladder_hf_ref.py):
    #   logits_at_step[i] = logits AT position (P-1+i), i.e. the
    #   distribution that would produce generated_ids[i] under greedy.
    #
    # So for i=0 (step 1) we need the logits after consuming prompt_ids
    # (i.e. last_logits from prefill). For i=1 we need logits after
    # consuming prompt_ids + generated_ids[0]. Etc.
    #
    # In our forward_token loop we call forward_token(token_id, cur_pos)
    # which RETURNS the logits at position cur_pos (i.e. the logits
    # used to predict the *next* token after cur_pos). So to populate
    # logits_at_step[i] we want forward_token(generated_ids[i-1] OR
    # prompt's last token, P-1+i).
    print(f"\nTeacher-forcing {M} steps over HF reference sequence…", flush=True)
    t_decode = time.time()

    # Prefill (P tokens, cur_pos 0..P-1)
    last_logits_tt = None
    for pos, tid in enumerate(prompt_ids):
        last_logits_tt = forward_token(tid, pos)
    ttnn.synchronize_device(device)
    print(f"  prefill done in {time.time()-t_decode:.1f}s", flush=True)

    # Per-step capture
    logits_tt_arr = np.empty((M, VOCAB), dtype=np.float32)
    # step 0: take last_logits_tt from prefill (= logits at pos P-1,
    # the prediction for generated_ids[0])
    logits_tt_arr[0] = ttnn.to_torch(last_logits_tt).float().cpu().numpy().flatten()

    # Now feed each HF-emitted token in turn and capture the resulting logits.
    # After feeding generated_ids[i-1] at position (P-1+i), the returned
    # logits are the prediction for token (P+i) — which corresponds to
    # logits_ref[i] (HF's step i).
    cur_pos = P
    t_steps = time.time()
    for i in range(1, M):
        # Feed generated_ids[i-1] at position (cur_pos = P + i - 1)
        t_s = time.time()
        last_logits_tt = forward_token(generated_ids[i - 1], cur_pos)
        ttnn.synchronize_device(device)
        logits_tt_arr[i] = ttnn.to_torch(last_logits_tt).float().cpu().numpy().flatten()
        cur_pos += 1
        if (i + 1) % 10 == 0 or i == M - 1:
            avg = (time.time() - t_steps) / (i)
            print(f"  step {i+1:3d}/{M}  pos={cur_pos-1}  "
                  f"({time.time()-t_s:.2f}s/step  avg {avg*1000:.0f}ms)",
                  flush=True)

    print(f"\nTT decode complete: {time.time()-t_decode:.1f}s", flush=True)

    # ----- Cosine ladder -----
    print(f"\n--- COSINE LADDER ---", flush=True)
    print(f"{'pos':>5s}  {'hf_top1_id':>10s}  {'tt_top1_id':>10s}  "
          f"{'match':>5s}  {'cos_logits':>12s}  "
          f"{'cos_top128':>12s}  {'top1_margin_hf':>14s}  "
          f"{'top1_margin_tt':>14s}", flush=True)
    print("-" * 100, flush=True)

    ladder_records = []
    requested = []
    for s in args.positions.split(","):
        s = s.strip()
        if not s:
            continue
        v = int(s)
        if 1 <= v <= M:
            requested.append(v)
    # Always compute full cosine array for full report
    per_pos_cos = np.empty(M, dtype=np.float64)
    per_pos_match = np.zeros(M, dtype=bool)
    for i in range(M):
        per_pos_cos[i] = cosine(logits_tt_arr[i], logits_ref[i])
        per_pos_match[i] = (int(logits_tt_arr[i].argmax()) ==
                             int(logits_ref[i].argmax()))

    for pos in requested:
        i = pos - 1  # 1-indexed
        hf_top1 = int(logits_ref[i].argmax())
        tt_top1 = int(logits_tt_arr[i].argmax())
        c = per_pos_cos[i]
        # Top-K cos over the most predictive logits (sort by HF magnitude)
        idx_top = np.argsort(-np.abs(logits_ref[i]))[:128]
        c128 = cosine(logits_tt_arr[i][idx_top], logits_ref[i][idx_top])
        sorted_hf = np.sort(logits_ref[i])[::-1]
        margin_hf = float(sorted_hf[0] - sorted_hf[1])
        sorted_tt = np.sort(logits_tt_arr[i])[::-1]
        margin_tt = float(sorted_tt[0] - sorted_tt[1])
        match = "Y" if hf_top1 == tt_top1 else "N"
        print(f"{pos:5d}  {hf_top1:10d}  {tt_top1:10d}  {match:>5s}  "
              f"{c:12.7f}  {c128:12.7f}  {margin_hf:14.4f}  {margin_tt:14.4f}",
              flush=True)
        ladder_records.append({
            "position": pos,
            "hf_top1_id": hf_top1,
            "tt_top1_id": tt_top1,
            "top1_match": bool(hf_top1 == tt_top1),
            "cos_full_vocab": float(c),
            "cos_top128_by_hf_abs": float(c128),
            "hf_top1_margin": margin_hf,
            "tt_top1_margin": margin_tt,
        })

    # First position where cos drops below thresholds
    thresholds = [0.999, 0.99, 0.9, 0.5]
    first_break = {}
    for thr in thresholds:
        below = np.where(per_pos_cos < thr)[0]
        first_break[str(thr)] = int(below[0] + 1) if len(below) else None
    print(f"\nFirst position where cos drops below threshold:", flush=True)
    for thr, pos in first_break.items():
        print(f"  cos < {thr}: position = {pos}", flush=True)

    # Top-1 match rate up to each requested position
    print(f"\nTop-1 match rate cumulative:", flush=True)
    for pos in requested:
        rate = float(per_pos_match[:pos].mean())
        print(f"  positions 1..{pos}: {int(per_pos_match[:pos].sum())}/{pos} = "
              f"{rate*100:.1f}% match", flush=True)

    # Quartile cosines for context
    print(f"\nCosine percentiles across all {M} positions:", flush=True)
    for q in [5, 25, 50, 75, 95]:
        v = float(np.percentile(per_pos_cos, q))
        print(f"  p{q}: {v:.7f}", flush=True)
    print(f"  min: {float(per_pos_cos.min()):.7f}  "
          f"max: {float(per_pos_cos.max()):.7f}", flush=True)

    # Save
    out = {
        "ref_dtype": ref_dtype,
        "prompt": prompt,
        "prompt_ids": prompt_ids,
        "M": M,
        "vocab": VOCAB,
        "ladder": ladder_records,
        "first_below_threshold": first_break,
        "per_pos_cos": per_pos_cos.tolist(),
        "per_pos_top1_match": per_pos_match.tolist(),
        "generated_ids_hf": generated_ids,
        "generated_ids_tt": [int(logits_tt_arr[i].argmax()) for i in range(M)],
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results → {args.out}", flush=True)


if __name__ == "__main__":
    main()
