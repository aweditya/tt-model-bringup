# MAX_POS Bump + Long-Form Generation Validation Plan — 2026-05-19

Today's qb2 production decode is locked at `MAX_POS = 512` (server_tp.py:50).
That cap means the entire (prompt + generation) chat must fit in 512 tokens —
unusable for a real coding-assistant workload, which routinely needs 8k-32k
contexts. This plan scopes the work to extend MAX_POS on qb2 TP and validate
real autoregressive long-form generation end-to-end.

## Today's state (HONEST)

| What | Status |
|---|---|
| Single-chip qb1 long context (paged SDPA) | **shipped** to MAX_POS=32k (`feedback_paged_sdpa_decode_works_at_32k.md`) |
| Single-chip qb1 needle-haystack at L=500 frac=0.5 | **passes** with B3 SDPA config (`feedback_fp32_sdpa_cliff_probe.md`) |
| Multi-chip qb2 TP MAX_POS | **512 cap** (`server_tp.py:50`) |
| qb2 TP teacher-forced cosine ladder at L=500 | **passes** (today's decay/gate G3) |
| qb2 TP **autoregressive** generation past 60 tokens | **never tested** |
| qb2 TP needle-haystack | **never ported from qb1** |
| qb2 TP coherence at 1k / 4k / 8k context | **unknown** |
| qb2 TP trace capture at MAX_POS > 512 | **never tested** |

The 500-token cosine ladders we've been running are *teacher-forced* (each
step receives the correct previous token from a CPU oracle and we just
compare logits) — they validate per-step *math correctness*, not real
generation. Production `generate_tp` has only ever been run at 60 tokens.

## What scales with MAX_POS

In `server_tp.py`:

| Constant | Today | Scales linearly with MAX_POS? | Notes |
|---|---|---|---|
| `MAX_POS = 512` | 512 | — | the cap |
| `NUM_BLOCKS = MAX_POS // 32` | 16 | yes | paged-cache blocks |
| KV cache per attn layer per chip | 0.27 MB | yes | bf16, sharded N_KV |
| `state.cos_table_tt`, `state.sin_table_tt` | [512, 64] | yes | RoPE lookup tables |
| Position-indexed buffers (`cur_pos_buf`, `rot_idxs_buf`) | scalar / [1] | no | per-token |
| SDPA k_chunk_size, q_chunk_size | program-config | indirectly | paged SDPA kernel tuning |
| Trace capture surfaces | one trace | no (one-time) | re-captured on first call |

KV cache memory math (16 attention layers × bf16):

| MAX_POS | Per-chip KV cache | Fits? (12 GB DRAM, 7 GB weights, 5 GB free) |
|---|---|---|
| 512 (today) | 4.2 MB | trivially |
| 1024 | 8.4 MB | trivially |
| 2048 | 16.8 MB | trivially |
| 4096 | 33.6 MB | trivially |
| 8192 | 67.1 MB | trivially |
| 16384 | 134 MB | comfortably |
| 32768 | 268 MB | comfortably |

**Memory is not the binding constraint.** The constraints are correctness +
trace capture + SDPA kernel behavior at large K.

## Risks to validate

1. **SDPA program config at long K.** The B3 HiFi2 config (commit reference in
   `feedback_paged_sdpa_shipped_tp.md`) was tuned at MAX_POS=256-512. At 8k,
   `k_chunk_size` and `q_chunk_size` may need re-tuning. Per
   `feedback_paged_sdpa_decode_works_at_32k.md` it works on single chip; on
   mesh paged SDPA needs `ttnn.SDPAProgramConfig(compute_with_storage_grid_size=ttnn.CoreCoord(4,4), q_chunk_size=0, k_chunk_size=0)` per `feedback_mesh_paged_sdpa_works.md`.

2. **Trace capture at long context.** Our trace was captured at MAX_POS=512.
   Bumping recaptures at first request. Possible failure modes:
   - L1 buffer sizing inside the kernel
   - Different program-config selection by SDPA at long K
   - Cos/sin table embedding lookups crossing tile boundaries differently

3. **Drift cliffs at long autoregressive context.** Per
   `feedback_bf16_prefill_drift_cliff.md`, qb1 had a cliff at position 129
   in bf16, FIXED by B3 HiFi2 config on the SDPA path. We use the same B3
   config on TP, so this should carry over — but our existing TP validation
   only goes to position 500 teacher-forced. Real autoregressive past
   ~129 has never been tested.

4. **DeltaNet recurrence state stability at long autoregressive context.**
   DeltaNet's recurrence state evolves every token. State magnitude growth
   under autoregressive feedback over thousands of tokens is unknown. The
   teacher-forced cosine ladder doesn't catch this because the inputs are
   reset from CPU each step.

5. **Generate-time host overhead.** Currently
   `generate_tp` does host-side argmax (8 bytes per token via vocab-sharded
   path). At 8k tokens × 80 ms = 640 sec wall — the user sees streaming
   tokens, so it's tolerable, but server response time matters.

## Bumping schedule (G-stages)

Same staged pattern as our successful kernel ships. Each gate either passes
(advance) or fails (root-cause + fix before next).

### G0 — Pre-flight (~1 hour, no bootstrap needed)

- Re-read code to confirm everything that touches MAX_POS auto-derives
  (NUM_BLOCKS, cos/sin tables, position buffers).
- Identify any constant that was hard-coded to 512 instead of MAX_POS.
- Confirm SDPA program-config code path works at K up to MAX_POS without
  assertion failures.
- Estimate trace capture wall (was ~85ms at 512; expect 1-2× longer at 8k).

### G1 — Smallest bump: MAX_POS=1024 (~1 bootstrap + 30 min testing)

- One-line edit: `MAX_POS = 1024`.
- Restart fresh server.
- Coherence test: `generate_tp --prompt "Write a Python function that..." --max-tokens 500`.
  - Pass: coherent technical text past 500 tokens.
  - Fail: gibberish or drift cliff → root-cause (most likely SDPA kernel config).
- Teacher-forced cosine ladder at 1000 positions vs HF bf16 oracle.
  - Pass: median cos ≥ 0.99, no cliff in 50-position rolling buckets.
  - Fail: localize where it diverges; compare to qb1's known cliffs.

### G2 — Needle-haystack port (~half-day)

- Port `experiments/utils/needle_haystack_b3_filter_run1.py` to qb2 TP path
  (uses `generate_tp` instead of single-chip `generate`).
- Run at L=1000, frac ∈ {0.25, 0.5, 0.75}.
- Pass: model returns the password verbatim at all 3 positions.
- Fail: same diagnosis path as `feedback_needle_haystack_qb1.md` — likely
  SDPA config, not architecture.

### G3 — MAX_POS=4096 (~1 bootstrap + 1 hour testing)

- Bump to 4096.
- Re-run G1 + G2 at 4000 positions.
- Measure tok/s at the larger context: KV cache reads grow → some latency
  hit expected. Memory note `feedback_kernel_profile_findings.md` would
  predict ~5-10% slowdown at 4k vs 512 from KV bandwidth scaling.

### G4 — MAX_POS=8192 (~1 bootstrap + 1 hour testing)

- Bump to 8192.
- Re-run G1 + G2.
- Generate a real coding query: "Write a tokenizer in Rust" with
  max_tokens=2000. Validate the output compiles / makes sense.

### G5 — MAX_POS=32768 ceiling test (~1 bootstrap + testing)

- Push to the natively-supported Qwen3.6 context (262k, but we'd start at
  32k as a realistic target).
- This is the "daily-driver code assistant" ceiling.
- If KV cache memory or kernel limits bite, document the wall.

## Deliverables per stage

For each Gn that passes:
- Commit + push the `MAX_POS = N` change
- Save artifacts in `.cache/qb2_tp_deltanet/` (cosine ladder npz, needle-haystack JSON, generate_tp transcript)
- Update HANDOFF.md "Current snapshot" with the new validated MAX_POS

For any Gn that fails:
- Root-cause before bumping further
- Document the failure mode in memory (so future agents don't repeat)

## Why this is the right next move

We've shipped multiple per-step tok/s wins (owned GDN, decay/gate, num_links=2)
totaling +7.1% session-over-session — but at the user's actual use case
(8k-context coding queries), our server returns an error. **MAX_POS is the
binding constraint for the product.**

After MAX_POS lands, the next product-meaningful work is HTTP frontend + a
thin VS Code shim. The vLLM-style continuous-batching / disaggregated PD path
is over-engineered for single-user daily-driver use; we can defer it.

## Effort estimate

| Stage | Wall-clock | Bootstraps |
|---|---|---|
| G0 pre-flight | 1 hr | 0 |
| G1 1024 | 1 hr + bootstrap | 1 |
| G2 needle port | half-day + bootstrap | 1 |
| G3 4096 | 2 hr + bootstrap | 1 |
| G4 8192 | 2 hr + bootstrap | 1 |
| G5 32768 (stretch) | 4 hr + bootstrap | 1 |

Total: **1-2 days of focused work + 5 bootstraps × 17 min = ~1.5 hours of bootstrap waits**.

## What we will NOT do in this plan

- Will not build continuous batching / disaggregated PD serving (out of scope; user said skip)
- Will not bump MAX_POS past 32k (no proven use case for daily-driver)
- Will not test prompts in arbitrary languages (English code queries only)
- Will not optimize tok/s at long context (correctness first; perf-at-context is its own follow-up)

## ADDENDUM 2026-05-19 — Prefill is the bigger gap

vLLM research agent flagged a CRITICAL finding that reshapes priorities:

**`handle_generate_tp` has NO real prefill.** It loops the decode trace once
per prompt token. At ~83 ms/tok:

| Prompt length | TTFT (time-to-first-token) |
|---|---|
| 100 tokens | 8.3 sec |
| 500 tokens | 42 sec |
| 1000 tokens | 83 sec |
| 4000 tokens | 5.5 min |
| 8000 tokens | 11 min |
| 32000 tokens | 44 min |

A real prefill kernel would process the entire prompt in ONE forward pass
with `seq_len = prompt_len` (compute-bound, batched SDPA). Without it, MAX_POS
extension only enables you to ASK longer questions, not actually USE them.

**Friend's `_prefill_forward_single_user`** at
`/Users/adityasriram/Labs/stanford/cs440lx/tt-xla/experiments/.refs/tt-qwen-36/models/tt_transformers/tt/generator.py`
shows the pattern. Per the build-from-scratch principle (`feedback_build_kernels_from_scratch.md`),
we'd build our own `forward_prefill_tp_inner` mirroring the structure of
`forward_token_tp_inner` but with `seq_len > 1`. Estimated effort: 3-5 days.

**Revised plan:**

- **Phase A (this plan as-is):** G0-G2 at MAX_POS ≤ 1024. Validates that the
  decode-only path is correct at longer contexts via teacher-forced + needle-
  haystack + autoregressive 500 tokens. ETA: 1 day. Still useful even with
  slow prefill — confirms the math doesn't cliff past 500 tokens
  autoregressive on TP.

- **Phase B (NEW):** Build real prefill kernel. `forward_prefill_tp_inner`
  with seq_len > 1, batched SDPA via `ttnn.transformer.scaled_dot_product_attention`
  (no `_decode` suffix; takes Q at [B, n_heads, seq_len, head_dim]).
  Separate trace for prefill. KV cache populated en masse. Verify TTFT
  at 1k = ~1 sec, 8k = ~5 sec. ETA: 3-5 days.

- **Phase C:** Bump MAX_POS to 8k+ AFTER prefill works. Re-run G3/G4 with
  real prefill timing. ETA: 1 day.

Total realistic effort for end-to-end usable 8k context: **~1 week** vs the
1-2 days originally estimated. The MAX_POS bump itself is still cheap; the
prefill kernel is the real engineering project.

**Order of operations recommendation:**
1. Do Phase A first (validates the AUTOREGRESSIVE correctness story, which
   is non-obvious — fail-fast on bf16 cliff at MAX_POS > 512)
2. If Phase A passes cleanly, build Phase B
3. Phase C is mechanical after B
