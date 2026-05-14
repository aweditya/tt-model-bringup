#!/usr/bin/env python3
"""P22 — vocab-sharded LM head + on-device argmax on (1,4) mesh (qb2).

Goal: validate that we can replace the production replicated-lm_head path
(server_tp.py:280, 692, 855-858) with a vocab-sharded matmul followed by an
on-device all_gather → untilize → argmax (per Agent X's resolution in
`feedback_lm_head_argmax_unknown.md` — friend's pattern at
`experiments/.refs/tt-qwen-36/models/common/sampling/tt_sampling.py:423-454`).

Compares:
  - PROD path: replicated lm_head linear → ConcatMeshToTensor(dim=0) → numpy argmax
  - NEW path:  sharded lm_head linear → all_gather_async(dim=-1)
               → untilize → ttnn.argmax(dim=-1) → tiny readback

Validates:
  1. Correctness: NEW argmax matches PROD argmax (and numpy gold) across 5 random hidden states.
  2. Latency: NEW total time vs PROD total time on 30-iter median.
  3. (Optional) cosine of gathered logits vs replicated logits, ≥ 0.999.

The matmul math is *the same numerically* (we just shard the weight along the
vocab axis). The win is:
  (a) ~4× less per-chip matmul work (~4.16 → ~1.05 ms projected)
  (b) eliminating the 152064 fp32 readback to host (~9.4 ms in Tracy)

Test shape: real Qwen3.6-27B lm_head [HIDDEN=5120, VOCAB=152064] bf16.

Pass gates (per `feedback_lm_head_argmax_unknown.md` follow-up):
  - argmax match on all 5 prompts
  - new path strictly faster than old path (≥ 2 ms/tok improvement)
"""
import json
import os
import sys
import time

import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)


VOCAB = 248320            # HF config vocab_size — matches state.embed_np.shape[0]
                          # (the old "152064" was the wrong value from stale notes;
                          # actual tokenizer vocab is 248044, model padded to 248320)
VOCAB_PADDED = 248320     # actual stored width of lm_head (= vocab_size)
HIDDEN = 5120
NCHIPS = 4
VOCAB_PER_CHIP_PADDED = VOCAB_PADDED // NCHIPS  # 62080 — what shards naturally
SEED = 7
N_TRIALS = 5
N_TIMING_ITERS = 30
OUTPUT_DIR = "/home/aditya/tt-xla/.cache/p22_vocab_sharded_lm_head"


def cosine(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 78)
    print("P22: vocab-sharded LM head + on-device argmax on (1,4) mesh (qb2)")
    print("=" * 78)
    print(f"weight shape: [{HIDDEN}, {VOCAB_PADDED}] bf16  (HF-padded; real vocab = {VOCAB})")
    print(f"per-chip padded slab: {VOCAB_PER_CHIP_PADDED} (= {VOCAB_PADDED}/{NCHIPS})  "
          f"tile-aligned: {'OK' if VOCAB_PER_CHIP_PADDED % 32 == 0 else 'NOT TILE-ALIGNED'}")
    print(f"real vocab tile-aligned: {'OK' if VOCAB % 32 == 0 else 'NOT TILE-ALIGNED'} ({VOCAB} / 32 = {VOCAB // 32})")

    print("\n[1] Loading real Qwen3.6-27B lm_head weight from HF…")
    # Use the same loader as 91l_fp32_residual_generate.py — returns pre-transposed [HIDDEN, VOCAB]
    sys.path.insert(0, "/home/aditya/tt-xla/experiments")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_91l", "/home/aditya/tt-xla/experiments/91l_fp32_residual_generate.py")
    _91l = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_91l)
    embed_weights = _91l.load_embed_lm_head_weights()
    W_np = embed_weights['lm_head']  # [HIDDEN, VOCAB_PADDED] fp32 (already transposed)
    assert W_np.shape == (HIDDEN, VOCAB_PADDED), f"unexpected lm_head shape {W_np.shape}"
    print(f"  ✓ lm_head shape: {W_np.shape}  range=[{W_np.min():.3f}, {W_np.max():.3f}]")

    print("\n[2] Init fabric + open (1,4) mesh…")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    print(f"  ✓ mesh {mesh.get_num_devices()} chips")

    results = {
        "shapes": {"hidden": HIDDEN, "vocab": VOCAB, "vocab_padded": VOCAB_PADDED,
                   "nchips": NCHIPS, "vocab_per_chip_padded": VOCAB_PER_CHIP_PADDED},
        "trials": [],
        "timing": {},
        "pass": False,
    }

    try:
        print("\n[3] Upload replicated lm_head (PROD path) — bf16…")
        t0 = time.time()
        W_repl_tt = ttnn.from_torch(
            torch.from_numpy(W_np),
            dtype=ttnn.bfloat16, device=mesh,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        ttnn.synchronize_device(mesh)
        print(f"  ✓ replicated upload took {time.time() - t0:.2f}s")

        print("\n[4] Upload sharded lm_head (NEW path) — bf16, dim=1 (vocab)…")
        t0 = time.time()
        W_sh_tt = ttnn.from_torch(
            torch.from_numpy(W_np),
            dtype=ttnn.bfloat16, device=mesh,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
        )
        ttnn.synchronize_device(mesh)
        print(f"  ✓ sharded upload took {time.time() - t0:.2f}s")

        print("\n[5] Correctness loop — 5 random hidden states …")
        rng = np.random.default_rng(SEED)
        all_match = True
        for trial in range(N_TRIALS):
            x_np = (rng.standard_normal((1, HIDDEN), dtype=np.float32) * 0.5).astype(np.float32)
            # numpy gold (fp32) — slice to real VOCAB to match production behavior
            y_gold = x_np @ W_np  # [1, VOCAB_PADDED]
            y_gold_real = y_gold[:, :VOCAB]
            gold_argmax = int(y_gold_real.argmax())

            x_tt = ttnn.from_torch(
                torch.from_numpy(x_np),
                dtype=ttnn.bfloat16, device=mesh,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            ttnn.synchronize_device(mesh)

            # --- PROD path (replicated matmul, host-side slice to real VOCAB, numpy argmax) ---
            y_repl_tt = ttnn.linear(x_tt, W_repl_tt)
            ttnn.synchronize_device(mesh)
            y_repl_np = ttnn.to_torch(
                y_repl_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
            ).float().cpu().numpy()
            # y_repl_np shape [NCHIPS, VOCAB_PADDED] (replicated → all rows equal). Pick chip 0,
            # then slice to real VOCAB (mirrors server_tp.py:857 [: state.embed_np.shape[0]]).
            y_repl_chip0_padded = y_repl_np[0].reshape(-1)
            y_repl_chip0 = y_repl_chip0_padded[: VOCAB]
            prod_argmax = int(np.argmax(y_repl_chip0))

            # --- NEW path: sharded linear → all_gather → slice → untilize → ttnn.argmax ---
            y_sh_tt = ttnn.linear(x_tt, W_sh_tt)  # per-chip [1, VOCAB_PER_CHIP_PADDED]=62080
            ttnn.synchronize_device(mesh)
            # all_gather along last dim → replicates the [1, VOCAB_PADDED] tensor on every chip
            y_gathered_tt = ttnn.all_gather(y_sh_tt, dim=-1)
            ttnn.synchronize_device(mesh)
            # Slice to real VOCAB on device (152064 is tile-aligned; the padding rows
            # would otherwise dominate argmax for random weights).
            # Shape after linear: rank-2 [1, 248320]. Slice last dim to 152064.
            y_sliced_tt = ttnn.slice(y_gathered_tt, [0, 0], [1, VOCAB])
            ttnn.synchronize_device(mesh)
            # untilize for argmax (per friend's pattern)
            y_rm_tt = ttnn.untilize(y_sliced_tt, use_multicore=True)
            ttnn.synchronize_device(mesh)
            # on-device argmax — uses keepdim=True + use_multicore=True (the
            # combo that works at vocab=152064; keepdim=False on this size
            # returns garbage bit-patterns in our ttnn build — see p22_argmax_sanity6).
            idx_tt = ttnn.argmax(y_rm_tt, dim=-1, keepdim=True, use_multicore=True)
            ttnn.synchronize_device(mesh)

            if trial == 0:
                print(f"  [diag] idx_tt: shape={tuple(idx_tt.shape)} dtype={idx_tt.dtype} layout={idx_tt.layout}")

            # Read back — result is replicated post-AG; each chip returns the
            # same idx. ConcatMeshToTensor(dim=0) → [NCHIPS, 1, 1]; pick chip 0.
            idx_concat = ttnn.to_torch(
                idx_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
            ).cpu().numpy()
            new_argmax = int(idx_concat.reshape(-1)[0])

            # Cosine of gathered logits (vs PROD) — should be bit-identical (same matmul math)
            y_gathered_np = ttnn.to_torch(
                y_gathered_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
            ).float().cpu().numpy()
            y_gathered_chip0 = y_gathered_np[0].reshape(-1)[: VOCAB]
            cos_vs_prod = cosine(y_gathered_chip0, y_repl_chip0)
            cos_vs_gold = cosine(y_gathered_chip0, y_gold_real.flatten())

            # The PRIMARY gate is: does NEW agree with PROD? PROD is already bf16
            # and is the current production behavior. NEW just shards the matmul.
            # If new != prod, we've changed behavior — RED. If new == prod (even
            # when both differ from fp32 gold due to bf16 razor-thin margins),
            # we're shippable.
            match_prod = (new_argmax == prod_argmax)
            match_gold = (new_argmax == gold_argmax)
            all_match = all_match and match_prod
            print(f"  trial {trial}: gold={gold_argmax:>6d}  prod={prod_argmax:>6d}  new={new_argmax:>6d}  "
                  f"cos(new,prod)={cos_vs_prod:.6f}  cos(new,gold)={cos_vs_gold:.6f}  "
                  f"new==prod={match_prod}  new==gold={match_gold}")
            results["trials"].append({
                "trial": trial,
                "gold_argmax": gold_argmax,
                "prod_argmax": prod_argmax,
                "new_argmax": new_argmax,
                "cos_vs_prod": cos_vs_prod,
                "cos_vs_gold": cos_vs_gold,
                "match_prod": match_prod,
                "match_gold": match_gold,
            })

            # Clean up per-trial tensors
            ttnn.deallocate(x_tt)
            ttnn.deallocate(y_repl_tt)
            ttnn.deallocate(y_sh_tt)
            ttnn.deallocate(y_gathered_tt)
            ttnn.deallocate(y_sliced_tt)
            ttnn.deallocate(y_rm_tt)
            ttnn.deallocate(idx_tt)

        n_match_prod = sum(t['match_prod'] for t in results['trials'])
        n_match_gold = sum(t['match_gold'] for t in results['trials'])
        print(f"\n  Correctness vs PROD (the gate): {'PASS' if all_match else 'FAIL'} ({n_match_prod}/{N_TRIALS} match)")
        print(f"  Correctness vs fp32 gold (informational, bf16 thin-margin): {n_match_gold}/{N_TRIALS}")

        # ---------- Timing benchmark ----------
        print(f"\n[6] Timing — {N_TIMING_ITERS} iters each path …")
        rng = np.random.default_rng(SEED + 99)
        x_np = (rng.standard_normal((1, HIDDEN), dtype=np.float32) * 0.5).astype(np.float32)
        x_tt = ttnn.from_torch(
            torch.from_numpy(x_np),
            dtype=ttnn.bfloat16, device=mesh,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        ttnn.synchronize_device(mesh)

        # PROD: replicated matmul + ConcatMeshToTensor readback + numpy argmax
        # Warmup
        _y = ttnn.linear(x_tt, W_repl_tt)
        ttnn.synchronize_device(mesh)
        _ = ttnn.to_torch(_y, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        ttnn.deallocate(_y)
        ttnn.synchronize_device(mesh)

        prod_times = []
        for _ in range(N_TIMING_ITERS):
            t0 = time.time()
            y = ttnn.linear(x_tt, W_repl_tt)
            y_np = ttnn.to_torch(y, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
            y_chip0 = y_np[0].float().cpu().numpy().reshape(-1)[: VOCAB]
            _ = int(np.argmax(y_chip0))
            ttnn.synchronize_device(mesh)
            prod_times.append((time.time() - t0) * 1000.0)
            ttnn.deallocate(y)
        prod_median_ms = float(np.median(prod_times))
        print(f"  PROD median: {prod_median_ms:.3f} ms   (p5={np.percentile(prod_times, 5):.3f}, p95={np.percentile(prod_times, 95):.3f})")

        # NEW: sharded matmul + all_gather + slice + untilize + argmax + tiny readback
        # Warmup
        _y = ttnn.linear(x_tt, W_sh_tt)
        _g = ttnn.all_gather(_y, dim=-1)
        _s = ttnn.slice(_g, [0, 0], [1, VOCAB])
        _u = ttnn.untilize(_s, use_multicore=True)
        _i = ttnn.argmax(_u, dim=-1, keepdim=True, use_multicore=True)
        ttnn.synchronize_device(mesh)
        ttnn.deallocate(_y); ttnn.deallocate(_g); ttnn.deallocate(_s); ttnn.deallocate(_u); ttnn.deallocate(_i)

        new_times = []
        for _ in range(N_TIMING_ITERS):
            t0 = time.time()
            y_sh = ttnn.linear(x_tt, W_sh_tt)
            y_g = ttnn.all_gather(y_sh, dim=-1)
            y_s = ttnn.slice(y_g, [0, 0], [1, VOCAB])
            y_u = ttnn.untilize(y_s, use_multicore=True)
            idx = ttnn.argmax(y_u, dim=-1, keepdim=True, use_multicore=True)
            idx_concat = ttnn.to_torch(idx, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
            _ = int(idx_concat.reshape(-1)[0])
            ttnn.synchronize_device(mesh)
            new_times.append((time.time() - t0) * 1000.0)
            ttnn.deallocate(y_sh); ttnn.deallocate(y_g); ttnn.deallocate(y_s); ttnn.deallocate(y_u); ttnn.deallocate(idx)
        new_median_ms = float(np.median(new_times))
        print(f"  NEW  median: {new_median_ms:.3f} ms   (p5={np.percentile(new_times, 5):.3f}, p95={np.percentile(new_times, 95):.3f})")

        delta_ms = prod_median_ms - new_median_ms
        speedup = prod_median_ms / new_median_ms if new_median_ms > 0 else float('inf')
        print(f"\n  Δ = {delta_ms:+.3f} ms  ({speedup:.2f}× speedup)")
        results["timing"] = {
            "prod_median_ms": prod_median_ms,
            "new_median_ms": new_median_ms,
            "prod_times_ms": prod_times,
            "new_times_ms": new_times,
            "delta_ms": delta_ms,
            "speedup_x": speedup,
        }

        # --- Verdict ---
        correctness_ok = all_match
        latency_improvement = delta_ms >= 2.0
        results["pass"] = bool(correctness_ok and latency_improvement)

        print("\n" + "=" * 78)
        print("VERDICT")
        print("=" * 78)
        print(f"  correctness (new == prod for all {N_TRIALS} trials): {'PASS' if correctness_ok else 'FAIL'}")
        print(f"  latency improvement ≥ 2 ms:                       "
              f"{'PASS' if latency_improvement else f'FAIL (Δ={delta_ms:+.3f} ms)'}")
        print(f"  OVERALL: {'PASS — ship to server_tp.py' if results['pass'] else 'FAIL — do NOT ship'}")

    finally:
        # Save results
        results_path = os.path.join(OUTPUT_DIR, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  ✓ wrote {results_path}")
        try:
            ttnn.close_mesh_device(mesh)
            print("  ✓ mesh closed cleanly")
        except Exception as e:
            print(f"  ✗ close error: {e}")
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            print("  ✓ fabric reset to DISABLED")
        except Exception as e:
            print(f"  ✗ fabric reset error: {e}")


if __name__ == "__main__":
    main()
