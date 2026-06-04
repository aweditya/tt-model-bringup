# 35B-A3B free-run greedy determinism (2026-06-04)

## Problem statement

Free-run greedy decode of Qwen3.6-35B-A3B on qb1 (1,4) P150 mesh is
**not bit-reproducible**: identical prompt + identical seed +
identical code produces different generated tokens on consecutive runs.
The needle-haystack measurement at `frac=0.5` shows ~50% retrieval rate
across `L ∈ {100, 200, 300, 460, 1024}` and individual trials flip
verdict (Y↔N) between runs ([[35b-needle-haystack-2026-06-04]]). Per
[[35b-drift-resolved-2026-06-04]] teacher-forced is now clean (cos ≥
0.987 at every probed position), so the floor is autoregressive
bf16 chain drift turning into a different argmax cascade.

This document catalogs the plausible sources, proposes localized
experiments, ranks mitigations, and gives a recommendation.

## 1. Catalog of non-determinism sources (ranked by likelihood)

1. **bf16 argmax tie-flip from non-associative reductions inside the
   forward.** Highest-prior. The 40-layer chain contains O(10²)
   reductions per layer (per-head sums in SDPA, all_reduce sums across
   four chips per attention/MoE block, expert-output sums in the MoE
   collector). Floating-point addition is non-associative; if any of
   those reductions don't pin a fixed visitation order across runs,
   identical inputs produce 1-ULP-different bf16 outputs. After 40
   layers the divergence is O(1e-5 .. 1e-4) — well below the cosine
   noise floor (cos ≈ 0.9999) but enough to flip a near-tie at the
   final lm_head argmax. The literature is unambiguous here: Liu et al.
   (Give Me FP32 or Give Me Death, 2026) measure **99.6%–100%**
   per-example divergence under bf16 greedy on DeepSeek-R1-Distill-Qwen
   on AIME/MATH500, dropping to ~0–5.8% under fp32. The 35B mechanism
   is the same — our 50% trial-flip rate is consistent with a model
   where a small fraction of decoded tokens are within 1 ULP of a tie
   and the chain seeds new ties downstream.

2. **DN recurrent state H_t evolution under bf16.** Once a single token
   diverges, the GatedDeltaNet recurrent state at L0..L31, L32..L47
   inherits a perturbed feature input and the recurrent multiply-add
   amplifies through subsequent steps. This is the standard "chaotic
   trajectory" effect for recurrences and matches the earlier per-step
   drift pattern in [[35b-a3b-l32-dn-decode-drift]] — though that
   memory's specific numbers were on the broken manual path
   ([[35b-manual-recurrence-path-broken]]) and now stale. This source
   is downstream of (1): it converts a ULP into a divergent trajectory,
   but does not by itself produce non-determinism between runs that
   start with identical state.

3. **`ttnn.all_reduce(cluster_axis=1)` reduce order.**
   `server_35b_ttnn.py:367-369` issues an unparameterised all_reduce
   per AT/MoE block (lines 725, 913, 1052, 1166, 1244, 1354). The
   public docs (`docs.tenstorrent.com/tt-metal/.../ttnn.all_reduce.html`)
   list a `topology` arg but make **no statement about reproducibility
   or fixed reduce order**. Ring-topology reductions in NCCL are not
   bit-deterministic across runs in general (CUDA-streams + worker
   scheduling can re-order); whether tt-metal's CCL pins the order is
   unknown to us. This is the single most likely non-determinism source
   that we can *test cheaply*.

4. **`ttnn.argmax(use_multicore=True)` tie behavior.** Line 1713 uses
   the multicore argmax. If two logits in `[1, VOCAB]` are bit-equal
   (a real possibility for early decode steps where logits cluster),
   different cores may "win" the reduction across runs depending on
   completion order. The single-core argmax would be deterministic by
   construction. This is also cheap to test.

5. **`ttnn.topk` ordering for ties.** When `_step_sampled_topk` is the
   active path (CB35 prod default per the topk-mode workaround for #149),
   any tie inside the top-K is sorted by an unspecified secondary key.
   At K=64 the prob mass is high so ties matter less, but the
   "index 0 == argmax" assumption in `cb_scheduler.py:676` is a
   tie-flip surface.

6. **Allocator / L1 placement evolution across `reset_caches_ttnn()`.**
   Speculative. If L1 placement of intermediate tensors shifts
   between runs (different allocator state, different per-core
   compute order), per-core sub-reductions can visit data in a
   different order. We have no evidence this happens, but `reset_caches_ttnn`
   does deallocate and recreate the DN/KV state on every reset
   (`server_35b_ttnn.py:1423-1455`). Easy to disprove: run twice
   without reset between, compare bit-exact.

7. **Async CCL.** `all_reduce_async` was tried and shelved
   ([[feedback-async-ccl-negative]]); the prod path uses synchronous
   `all_reduce`. Not currently a source — but if anyone re-enables
   async, it joins the suspect list.

## 2. Localization experiments (in dev harness)

All of these are 30-second iterations on the cb35 dev harness
([[reference_gm4_dev_harness.md]] for the launcher pattern); none
require server restart. Each is gated and small.

- **E1: same forward, twice, no reset.** Capture per-layer
  `_ttnn_to_numpy_replicated(h_tt, mesh)` for L0..L40 across two
  back-to-back identical forwards (single position, fixed prompt).
  Use existing per-layer hooks in `cb35_per_layer_drift_pos1.py`.
  → If bit-equal: non-determinism is *between forwards*, isolating to
  reset/allocator (source 6) or external scheduler state. If different:
  non-determinism is *within one forward*, isolating to one of (3),
  (4), or upstream parallel reductions.

- **E2: divergence step index.** Run two independent free-run decodes
  on the same prompt+seed, compare token-by-token. Note `K` =
  first divergent step. Then capture top-5 logits at step `K` in run 1
  and at step `K` in run 2: report `|logit[top1] - logit[top2]|` in
  ULPs and bytes. If the gap is ≤1 ULP, source (1) is confirmed. If
  ≥10 ULP, there's a real non-deterministic op upstream.

- **E3: ablate all_reduce.** Wrap `all_reduce_tt` to optionally do a
  CPU readback + numpy sum + write-back (slow, deterministic). Same
  prompt × N. If output stabilizes → all_reduce (source 3) is the
  primary source. ~30 lines, runs in eager.

- **E4: ablate argmax.** Toggle `ttnn.argmax(use_multicore=False)`
  OR replace with `np.argmax` of a full logits readback. Same
  prompt × N. If output stabilizes → multicore argmax (source 4).

- **E5: top-1 host argmax with explicit tie-break.** Even if the
  underlying logits differ by 1 ULP across runs, this is a no-op for
  argmax UNLESS the rank-1 element changes. Implement
  `argmax-then-resolve-tie-by-lowest-token-id` on host (vectorized
  on `logits[s]` in `_step_sampled_logits`). Same prompt × N. If
  output stabilizes → confirms ties (source 1) AND ships a fix.

- **E6: fp32 free-run feasibility.** `logits.float()` already exists
  in `cb_scheduler.py:621`. Run E5 with the final matmul (lm_head) at
  fp32 dest accumulate (HIFI4 with `fp32_dest_acc=True`). If output
  stabilizes → upstream chain accumulates ULP noise but the lm_head
  is the tie-flip surface; cheap to harden.

Order: E1 → E2 → (E3 ∥ E4 ∥ E5). Skip E6 unless E5 is insufficient.

## 3. Mitigation options

| # | Option | Cost | Effect | Notes |
|---|---|---|---|---|
| A | **Top-1 with deterministic tie-break by lowest token-id** | ~20 LOC in `cb_scheduler._step_sampled_logits` and `_step_sampled_topk` | Eliminates argmax-flip non-determinism even if underlying bf16 logits drift up to 1 ULP. Does NOT fix divergence if logits drift >1 ULP. | Drop-in; bf16-safe; matches HuggingFace `do_sample=False` tie-break convention. |
| B | **Force single-core argmax** | 1 char (`use_multicore=False`) | Removes source 4. Negligible perf at VOCAB=248320 since the readback dominates. | Try alongside A. |
| C | **Stabilize all_reduce order** | unknown — need ttnn docs / dialog with Tenstorrent | Removes source 3 if it's real. | Requires E3 to confirm worth pursuing. |
| D | **`fp32_dest_acc=True` on lm_head matmul only** | Half a line | Reduces ULP noise at the final logits. Cheapest precision lever. | Pair with A. |
| E | **Top-k sampling at temperature ε with fixed PRNG** | ~3 LOC sampling config | "User-visible" deterministic: same seed → same output. Hides the underlying non-determinism rather than fixing it. | Acceptable for chat; weird for benchmarks. |
| F | **fp32 H_t on DN path** | Requires fixing [[35b-manual-recurrence-path-broken]] first | Reduces recurrent amplification but doesn't touch the source (1) attention path. | Long-tail; not a session-shot. |
| G | **LayerCast (Liu et al.): fp32 compute, bf16 storage** | Major: change every matmul `compute_kernel_config` to fp32 dest_acc + careful dest fitting | Reportedly drops bf16 divergence rate to ~fp32 (≤5.8%) at 34% memory cost vs full fp32. | Multi-week. Not a sane next step. |
| H | **Full fp32 weights + activations** | DRAM-busting (40 layers × ~4 GB/chip ≈ 16 GB/chip; we have 31.83 GB/chip per [[feedback-p150-memory-bandwidth-measured]]) | Eliminates source 1 by construction. | Possibly feasible at 35B since weights fit; perf cost is severe. |

## 4. How other frameworks handle this

- **vLLM**: `temperature=0.0` is officially *not* bit-reproducible.
  Issues #608, #1202, #15437 are all closed-as-not-planned. The
  maintainer position is that bf16 + parallel reductions + dynamic
  batching cannot be made deterministic at acceptable cost.
- **PyTorch + CUDA**: `torch.use_deterministic_algorithms(True)` and
  `torch.backends.cudnn.deterministic=True` reduce sources but **do
  not promise bit-determinism for bf16 reductions on multiple SMs**;
  the docs explicitly note `scatter_add` / `index_add` are still
  non-deterministic at bf16. Standard recommendation: fp32.
- **Recent research** (Liu et al., *Give Me FP32 or Give Me Death?*,
  arXiv 2506.09501v2): identifies non-associative parallel reductions
  + bf16 mantissa (7 bits) as the dominant source; measures 99.6–100%
  example-level divergence under bf16 greedy on DeepSeek-R1-Distill-Qwen,
  dropping to ≤5.8% under fp32; proposes **LayerCast** (bf16 storage,
  fp32 compute, JIT cast on matmul) for 34% memory savings vs pure
  fp32.
- **TT-Metal**: `ttnn.all_reduce` documentation makes **no
  reproducibility guarantee**. We do not know whether the underlying
  CCL kernel pins reduce order; recommend confirming via E3 before
  assuming.

## 5. Recommendation

Short-term (this session, ~1 hour):

1. **Ship A + B + D together.** A: deterministic tie-break by
   lowest-id in `_step_sampled_logits` and `_step_sampled_topk`. B:
   `ttnn.argmax(use_multicore=False)` (negligible perf, removes one
   source). D: `fp32_dest_acc=True` on the lm_head matmul only. Each
   touches one file, each is reversible behind an env flag.
2. **Re-run needle-haystack 3× per `L`.** If retrieval flips 0/3 →
   shipped. If still flips → run E1, E2.

Medium-term (next session):

3. Run E1 → E2 → E3/E4/E5. Localize the dominant source. If it's
   all_reduce (E3), pursue option C with Tenstorrent on a topology
   that pins order; if it's near-ties (E2 confirms ≤1 ULP gaps),
   document A+B+D as the production fix and move on.
4. If user-visible determinism is required for benchmarks, switch
   the bench harness to **fixed-seed top-1 sampling with
   tie-break by id** (option A) — this is already the recommended
   practice in the upstream literature.

Long-term (post-task #164):

5. Once [[35b-manual-recurrence-path-broken]] is closed, A/B fp32
   H_t (option F) on the DN path. Expect a moderate (not dominant)
   reduction in trajectory divergence.
6. **Do not pursue G or H** unless a downstream customer explicitly
   demands strict bit-reproducibility at long context. Per
   [[bf16-chain-drift-at-B-gt-1]] this is a fundamental precision
   floor and the right framing is "use cosine, not exact-token-match,
   for bench/profile."

## Honest limits of this analysis

- We have **not** confirmed that `ttnn.all_reduce` is the actual
  reorder source. The docs are silent; only E3 settles it.
- The 1-ULP-tie hypothesis is consistent with the 50% trial-flip
  rate but unverified at the per-step level. E2 settles it.
- Source 6 (allocator) is speculative and listed last because we
  have no positive evidence — only an "if E1 fails, look here" note.
- The fp32-lm_head idea (D) helps only the final tie-flip; it does
  nothing about per-layer drift sources (1) (3).

## Related memory

[[35b-needle-haystack-2026-06-04]],
[[35b-drift-resolved-2026-06-04]],
[[bf16-chain-drift-at-B-gt-1]],
[[35b-a3b-l32-dn-decode-drift]],
[[35b-a3b-drift-is-user-facing]],
[[35b-manual-recurrence-path-broken]],
[[feedback-async-ccl-negative]],
[[feedback-cos-not-enough-also-check-mad]],
[[reference-p150-roofline-priority]],
[[feedback-p150-memory-bandwidth-measured]].

## Sources

- Liu et al., *Give Me FP32 or Give Me Death? Challenges and Solutions
  for Reproducible Reasoning*, arXiv 2506.09501v2
  (https://arxiv.org/html/2506.09501v2).
- vLLM issues #608, #1202, #15437 (all closed as not planned).
- TT-NN docs: `ttnn.all_reduce`, `ttnn.argmax`, `ttnn.topk`
  (https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api/).
