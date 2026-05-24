#!/usr/bin/env python3
"""Teacher-forced cosine ladder for Qwen3.6-35B-A3B on (1,4) qb1 mesh.

For each prompt position pos:
  - feed prompt_ids[pos] to step_forward_ttnn with capture={}
  - extract per-layer hidden states + final norm + logits
  - cosine-compare to HF oracle at same pos (`.cache/hf_oracle_35b*/`)
  - record cosine curve + top-1 match + layer-of-first-divergence

This is the 35B-A3B analog of the 27B `cosine_ladder` endpoint that found
the bf16 SDPA HiFi4 cliff at position 129 (see `feedback_fp32_sdpa_cliff_probe.md`).
Standalone probe; promote to a `cosine_ladder` endpoint on
`server_35b_ttnn.py` once the shape stabilizes.

Run (qb1, ttnn env exported, server stopped):
  cd ~/tt-xla
  .venv/bin/python -u experiments/utils/cosine_ladder_35b.py \\
    --oracle .cache/hf_oracle_35b_100tok \\
    --output-json .cache/sanity_2026_05_22/cosine_ladder_35b.json \\
    [--n-positions 10]   # smoke; omit for full prompt

Output JSON schema documented in `research/35b_a3b_correctness_plan.md`.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))
sys.stdout.reconfigure(line_buffering=True)  # SSH pipes are block-buffered

import server_35b_ttnn as srv  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cos(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True,
                    help="HF oracle dir (must contain hidden_states.npy, logits.npy, argmax.npy, prompt_ids.npy, meta.json)")
    ap.add_argument("--output-json", required=True,
                    help="path to write per-position results JSON")
    ap.add_argument("--n-positions", type=int, default=0,
                    help="cap at first N positions (0 = full prompt). Use small N for smoke runs.")
    ap.add_argument("--drift-cos-threshold", type=float, default=0.99,
                    help="layer cosine below this counts as divergence")
    ap.add_argument("--attn-mode", choices=["manual", "sdpa"], default="manual",
                    help="attention path: 'manual' (B16/B17 default) or 'sdpa' (paged + B3 config)")
    ap.add_argument("--no-rope-broadcast", action="store_true",
                    help="(sdpa mode only) skip the K-broadcast workaround; apply RoPE directly on [1, HEAD_DIM]. "
                         "Phase 2 ablation — if ttnn [1, HEAD_DIM] slice bug is still live, K will be corrupted.")
    ap.add_argument("--capture-attn-layer", type=int, default=None,
                    help="(sdpa mode only) capture attn sub-step intermediates at this decoder layer "
                         "index (0..n_layers-1). Must be a full_attention layer. Saves npz alongside JSON.")
    ap.add_argument("--sdpa-variant", choices=["B3", "hifi4_fp32"], default="B3",
                    help="(sdpa mode only) compute_kernel_config variant for paged SDPA. "
                         "B3 = 27B-validated recipe; hifi4_fp32 = HiFi4 + fp32_dest_acc.")
    ap.add_argument("--kv-cache-dtype", choices=["bf16", "fp32"], default="bf16",
                    help="(sdpa mode only) KV cache storage dtype. fp32 was hard-rejected by "
                         "paged SDPA decode in 27B's ttnn build — may have been fixed since.")
    ap.add_argument("--residual-add-dtype", choices=["bf16", "fp32"], default="bf16",
                    help="dtype for the residual ADDs in layer_forward_ttnn. fp32 = upcast "
                         "before each ttnn.add, cast back to bf16 after — mitigates bf16 "
                         "quantization noise on large-magnitude tensors at late layers.")
    args = ap.parse_args()

    oracle_dir = Path(args.oracle)
    hf_hidden_states = np.load(oracle_dir / "hidden_states.npy")  # [41, seq, HIDDEN]
    hf_logits = np.load(oracle_dir / "logits.npy")  # [seq, VOCAB]
    hf_argmax = np.load(oracle_dir / "argmax.npy")  # [seq]
    hf_prompt_ids = np.load(oracle_dir / "prompt_ids.npy")  # [seq]
    meta = json.loads((oracle_dir / "meta.json").read_text())
    n_layers_plus_1 = hf_hidden_states.shape[0]  # 41 for 40-layer model
    n_layers = n_layers_plus_1 - 1
    seq_len_full = hf_prompt_ids.shape[0]
    n_positions = seq_len_full if args.n_positions == 0 else min(args.n_positions, seq_len_full)
    log(f"oracle: {oracle_dir} | seq_len={seq_len_full} | layers={n_layers} | positions to test: {n_positions}")
    log(f"prompt: {meta['prompt'][:80]!r}{'…' if len(meta['prompt']) > 80 else ''}")

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Bootstrap (1,4) mesh + load all 40 layer weights. ~106 s on qb1 per
    # B16 smoke timings.
    log(f"bootstrapping (1,4) mesh + uploading weights… "
        f"(attn_mode={args.attn_mode}, no_rope_broadcast={args.no_rope_broadcast})")
    state = srv.State()
    state.attn_mode = args.attn_mode  # MUST be set before bootstrap; allocates paged plumbing if sdpa
    state.attn_sdpa_no_broadcast = args.no_rope_broadcast
    state.sdpa_compute_variant = args.sdpa_variant
    state.kv_cache_dtype = args.kv_cache_dtype
    state.residual_add_dtype = args.residual_add_dtype
    t0 = time.time()
    srv.bootstrap(state, log)
    state.reset_caches_ttnn()  # zero DN conv/recurrent caches + KV cache placeholders (paged if sdpa)
    log(f"bootstrap done in {time.time()-t0:.1f}s")

    # Warmup step + sync before the actual ladder. Suspicion: SDPA mode shows
    # pos-0-only corruption that may be a JIT-compile race on first call.
    # Run a single forward + sync + reset caches to populate JIT caches.
    import ttnn as _ttnn  # local import — keep probe top clean
    log("warmup: 1 forward call to populate JIT caches, then sync + reset…")
    warm_t0 = time.time()
    _ = srv.step_forward_ttnn(state, int(hf_prompt_ids[0]), 0)
    _ttnn.synchronize_device(state.mesh)
    state.reset_caches_ttnn()  # reset KV/DN caches after warmup
    log(f"warmup + sync + reset done in {time.time()-warm_t0:.1f}s")

    per_pos = []
    first_div_pos = None
    first_div_layer = None
    # When --capture-attn-layer is set, request sub_capture for that layer;
    # store the per-position attn sub-step arrays in a dict to npz-dump after.
    attn_sub_by_pos = {}
    for pos in range(n_positions):
        tok_id = int(hf_prompt_ids[pos])
        cap = {}
        if args.capture_attn_layer is not None:
            cap["sub_capture_layers"] = [args.capture_attn_layer]
        t0 = time.time()
        tt_next_id = srv.step_forward_ttnn(state, tok_id, pos, capture=cap)
        step_ms = (time.time() - t0) * 1e3
        if args.capture_attn_layer is not None:
            L_sub = cap.get(f"layer_{args.capture_attn_layer}_sub", {})
            # Top-level layer captures: in_norm, mixer_out, after_mixer, post_attn_norm, moe_out
            # (saved by layer_forward_ttnn as direct keys in sub_capture dict).
            for k, v in L_sub.items():
                if isinstance(v, dict):
                    continue  # attn_sub / moe_sub handled below
                attn_sub_by_pos[f"pos{pos:03d}_layer_{k}"] = v
            for sub_dict_name in ("attn_sub", "moe_sub"):
                sd = L_sub.get(sub_dict_name, {})
                for k, v in sd.items():
                    attn_sub_by_pos[f"pos{pos:03d}_{k}"] = v

        # Per-layer cosine: embed (oracle idx 0) + each layer (oracle idx L+1)
        cos_per_layer = [cos(cap["embed"], hf_hidden_states[0, pos])]
        for L in range(n_layers):
            cos_per_layer.append(cos(cap[f"layer_{L}"], hf_hidden_states[L + 1, pos]))
        cos_final = cos(cap["final_norm"], hf_hidden_states[-1, pos])
        cos_logits = cos(cap["logits"], hf_logits[pos])

        hf_arg = int(hf_argmax[pos])
        top1 = (tt_next_id == hf_arg)

        # Record layer-of-first-divergence (first layer where cos < threshold)
        # for the EARLIEST position that diverges.
        if first_div_pos is None:
            for li, c in enumerate(cos_per_layer):
                if c < args.drift_cos_threshold:
                    first_div_pos = pos
                    first_div_layer = li  # 0 = embed, 1..40 = decoder layers
                    log(f"  ** first divergence at pos {pos} layer {li} (cos {c:.4f})")
                    break

        per_pos.append({
            "pos": pos,
            "tok_id": tok_id,
            "hf_argmax": hf_arg,
            "tt_argmax": tt_next_id,
            "top1_match": bool(top1),
            "cos_per_layer": [round(c, 6) for c in cos_per_layer],
            "cos_final_norm": round(cos_final, 6),
            "cos_logits": round(cos_logits, 6),
            "step_ms": round(step_ms, 1),
        })
        log(f"  pos={pos:3d} tok={tok_id:6d} tt={tt_next_id:6d} hf={hf_arg:6d} "
            f"match={'Y' if top1 else 'N'} cos_final={cos_final:.4f} "
            f"cos_min_layer={min(cos_per_layer):.4f} step={step_ms:.0f}ms")

    median_final = float(np.median([p["cos_final_norm"] for p in per_pos]))
    median_logits = float(np.median([p["cos_logits"] for p in per_pos]))
    top1_count = sum(1 for p in per_pos if p["top1_match"])

    out = {
        "oracle_dir": str(oracle_dir),
        "n_positions_tested": n_positions,
        "n_layers": n_layers,
        "seq_len_full": seq_len_full,
        "prompt": meta["prompt"],
        "prompt_ids": [int(x) for x in hf_prompt_ids[:n_positions]],
        "drift_cos_threshold": args.drift_cos_threshold,
        "first_divergence_pos": first_div_pos,
        "first_divergence_layer": first_div_layer,
        "median_cos_final_norm": median_final,
        "median_cos_logits": median_logits,
        "top1_match_count": top1_count,
        "top1_match_rate": top1_count / max(1, n_positions),
        "per_pos": per_pos,
    }
    out_path.write_text(json.dumps(out, indent=2))
    log(f"wrote {out_path}")
    if attn_sub_by_pos:
        npz_path = out_path.with_suffix(".attn_sub.npz")
        np.savez(npz_path, **attn_sub_by_pos)
        log(f"wrote {npz_path} ({len(attn_sub_by_pos)} arrays for layer {args.capture_attn_layer})")
    log(f"summary: top1 {top1_count}/{n_positions} ({100*top1_count/max(1,n_positions):.1f}%)  "
        f"median cos_final {median_final:.4f}  median cos_logits {median_logits:.4f}  "
        f"first_div: pos={first_div_pos} layer={first_div_layer}")


if __name__ == "__main__":
    main()
