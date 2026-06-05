# tt-metal adoption plan — what to pull in while Nemotron G1 lands

**Date**: 2026-06-04 (post-poster demo, post-audit).
**Status**: living plan.

## 0. Context

The 3 background audits (`research/audit_*.md`) surfaced ~10 concrete
techniques on the Tenstorrent side we could adopt into our server stack.
The user explicitly asked to schedule this work as **background tasks
running in parallel with the foreground Nemotron G1 kernel build**.

The Mamba2 SSD kernel takes a focused ~5 days (per
`research/mm7_g1_mamba2_kernel_design.md` §8). During that window, we
have idle qb1 hardware after each kernel iteration — perfect for
running validation experiments on the existing 27B / 35B / Gemma 4
servers.

This doc:
1. Ranks every adoption candidate from the 3 audits by ROI/hour.
2. Marks each as **NOW** (single-subagent task, fits inside the G1
   window), **NEXT** (substantial; should follow Nemotron), or
   **PARK** (low-ROI or duplicative).
3. Creates an explicit task list.

> **Non-negotiables compliance**: each adoption follows the recipe —
> plan first (cite the audit + source URL), permanent files, remote-only
> execution, frequent commits, reuse mandate (every borrowed pattern
> cites the upstream URL in its commit message).

------------------------------------------------------------------------

## 1. Ranking — all 10 candidates, scored

Effort: ★ = ≤2 hours, ★★ = 0.5-1 day, ★★★ = 2-5 days, ★★★★ = >1 week
Impact: ★ = nice-to-have, ★★ = measurable perf or correctness win, ★★★ = headline

| # | Source | Candidate | Effort | Impact | Tier |
|---|---|---|---|---|---|
| 1 | gm4 audit | `paged_fused_update_cache` (#44946) | ★ | ★★ | **NOW** |
| 2 | gm4 audit | Redundant `to_memory_config` / `to_layout` audit (#44958) | ★ | ★ | **NOW** |
| 3 | gdn audit | Tracy bake-off: `recurrent_gated_delta_rule_decode_ttnn` (changh95 branch) vs our `qwen36_gdn_decode_owned` | ★★ | ★★ | **NEXT-A** |
| 4 | gm4 audit | RMSNorm fusion (#44948) — distributed-RMSNorm + matmul fusion | ★★★ | ★★★ | **NEXT-B** |
| 5 | qwen audit | Chunk-outer 2048-tok prefill trace | ★★★ | ★★★ | **NEXT-C** |
| 6 | qwen audit | Masked fixed-bucket prefill {128, 256, 512, 1024, 2048} | ★★★ | ★★ | **NEXT-D** |
| 7 | gdn audit | Wider fused-op signature: absorb b_proj/a_proj/dt_bias/A_exp/z_proj/norm_weight into `deltanet_recurrence` | ★★★★ | ★★★ | **NEXT-E** |
| 8 | gdn audit | TT-LANG ablation `gdn_step_8head` (~100 LOC Python DSL vs our 540 LOC LLK) | ★★★★ | ★ | **PARK** |
| 9 | gdn audit | Trace-compat `ttnn.gather`-based head-sharding for B>1 CB | ★★★ | ★ | **PARK** |
| 10 | qwen audit | Ask Tenstorrent directly about "256K" claim (their code is 128K) | ★ | ★ | comms only |

### Notes on the ranking

- The two **NOW** items (1, 2) fit cleanly inside the G1 build window
  without scope creep. They both touch the Gemma 4 server only, leaving
  the Mamba2 kernel work untouched.
- **NEXT-A** (kernel bake-off, item 3) is borderline NOW — could go in
  the G1 window if the bake-off harness already exists. Worth a quick
  scoping check; if the gdn benchmark file can be retargeted in ~2
  hours, promote to NOW.
- **NEXT-B..E** are too big to one-shot. They get their own plan-of-
  action docs (in a follow-up) and should be scheduled after Nemotron
  Phase 0 (G4) lands. NEXT-B (RMSNorm fusion) is the biggest perf bet
  (+12-15 ms/tok projected per the briefing).
- **PARK** items are low expected ROI: item 8 (TT-LANG) is a learning
  exercise more than a perf bet; item 9 (trace-compat gather) only
  matters for irregular CB routing which we don't need yet.
- Item 10 is purely a comms ask — send the email reply already drafted
  in `research/audit_qwen36_us_vs_qwen9b_p150_branch.md` §6.

------------------------------------------------------------------------

## 2. NOW batch — exact scope for the background subagent

A single background subagent picks up items 1 + 2 against the Gemma 4
12B IT server backend. The Mamba2 kernel work continues in the
foreground.

### Item 1 — `paged_fused_update_cache` on Gemma 4 sliding layers

- **Source**: tt-metal issue #44946 +  `experiments/serve/server_gemma4_unified_ttnn.py`'s
  sliding-window path. We currently call `paged_update_cache` *twice*
  per sliding layer because `NKV_PER_CHIP=1` after the split (commit
  `e2ae9f2`).
- **Goal**: collapse the two calls into one fused
  `paged_fused_update_cache` invocation.
- **Projected win**: ~1.6 ms/tok (47.5 → 45.9 ms/tok target).
- **Gate**: existing Gemma 4 multi-turn HTTP smoke
  (`stress_multiturn_http.py --model google/gemma-4-12B-it`) must
  still pass; `cb_prefix_cache` metrics unchanged. Tracy measurement
  before/after must show the per-layer kernel call count reduced.

### Item 2 — Redundant `to_memory_config` / `to_layout` audit

- **Source**: tt-metal issue #44958. Same audit pattern that surfaced
  the `cb_dn_recurrence_mode` regression (`feedback_cb_api_clobbered_27b_owned_gdn`).
- **Goal**: grep `experiments/serve/server_gemma4_unified_ttnn.py`
  for every `to_memory_config` / `to_layout` call site; classify each
  as "necessary" or "redundant"; remove the redundant ones.
- **Projected win**: small per-call savings × many call sites; likely
  ~0.5-1 ms/tok aggregate.
- **Gate**: same Gemma 4 multi-turn smoke; cosine ladder unchanged
  vs pre-edit baseline.

------------------------------------------------------------------------

## 2a. Adoption results 2026-06-04

### Task #191 — `paged_fused_update_cache` — BLOCKED (input-overlap contract)

**Outcome**: REVERTED. The op exists in our tt-metal build
(`ttnn.experimental.paged_fused_update_cache` — `paged_cache.hpp:23`,
`paged_cache_nanobind.cpp:67`) but its device-op asserts that
`input_tensor1` and `input_tensor2` shard grids **must not overlap**
(`paged_fused_update_cache_device_operation.cpp:226`).

Our `_shard_for_paged_write{,_b}` puts both K and V on the same
`state.paged_write_mem_cfg_sliding` / `state.paged_write_mem_cfg_global`
core grid (32 cores: `{[(x=0,y=0)-(x=10,y=1)], [(x=0,y=2)-(x=9,y=2)]}`),
so the fused op raises `TT_FATAL: is_overlap` on first call. Server
crashed during bootstrap.

The 4 call sites swap was clean — single-stream sliding (line 1139-1148),
single-stream global (1233-1242), CB sliding (323-328), CB global
(394-399). All reverted to the original 2-call pattern.

**To unblock**: build a second HEIGHT_SHARDED L1 mem_cfg on a disjoint
core grid for the V shard. The 32-core grid in question is partially
filled (11×2 + 10 = 32 in cols 0-10 of rows 0-1 + cols 0-9 of row 2);
a clean disjoint split for [1, 1, BLOCK_SIZE, head_dim] (which fits in
1 core for NKV=1) would put K on core (0,0) and V on core (1,0). This
is a ~30 LOC change to `setup_state` (build a paired
`paged_write_mem_cfg_*_kv` pair instead of one) plus updating
`_shard_for_paged_write{,_b}` callers to pass the right cfg.

**Effort to actually land**: 2-4 hours including disjoint-cfg build,
contract verification via small probe (fork from
`experiments/cb/isolate/paged_update_cache.py`), then deploy +
multi-turn smoke. Bumped from NOW → NEXT-F.

### Task #192 — `to_memory_config` / `to_layout` audit — AUDIT-ONLY (no shipped removal)

**Outcome**: AUDIT DONE, NO REDUNDANT CALLS FOUND.

Enumeration of every `to_memory_config` / `to_layout` site in both
servers:

| File:line | Call | Classification |
|---|---|---|
| `server_gemma4_unified_ttnn.py:623` | `to_layout(embed, TILE)` after `ttnn.embedding` | Necessary — embedding returns ROW_MAJOR; rms_norm needs TILE |
| `server_gemma4_unified_ttnn.py:737` | `to_layout(embed2, TILE)` post-embedding | Necessary — same reason |
| `server_gemma4_unified_ttnn.py:1035-1036` | `to_layout(cos_row/sin_row, TILE)` post-embedding | Necessary — RoPE table is RM after lookup; elementwise wants TILE |
| `server_gemma4_unified_ttnn.py:1049,1058,1061` | `_shard_for_paged_write`: RM → reshape → pad → TILE → HEIGHT_SHARDED | All necessary — explicit layout pipeline mandated by `paged_update_cache` contract |
| `server_gemma4_unified_ttnn.py:1369,1411,1488` | Post-embed `to_layout(.., TILE)` × `EMBED_SCALE` | Necessary (same as 737) |
| `server_gemma4_unified_cb.py:266,271,273` | CB `_shard_for_paged_write_b`: RM → reshape → pad → TILE → HEIGHT_SHARDED | Necessary |
| `server_gemma4_unified_cb.py:303-304, 381-382` | CB sliding/global RoPE `to_layout` post-`ttnn.embedding` lookup | Necessary |
| `server_gemma4_unified_cb.py:467` | CB top-of-forward `to_layout(embed, TILE)` × `EMBED_SCALE` | Necessary |

Total call sites enumerated: **17** across both files. Of these, **0**
are unambiguously redundant — every site is a real layout/memory-config
transition between an `ttnn.embedding` (row-major output) and a
downstream tile-mode consumer, or part of the explicit `paged_update_cache`
shard pipeline.

The actual decode hot path has no spurious round-trips. The Tenstorrent
audit category #44958 doesn't fire on our codebase. Likely savings
< 0.5 ms/tok (probably < 0.1 ms/tok), insufficient to risk a wrong-fix
regression.

**Future work**: a Tracy capture on a steady-state decode iteration
might still reveal sub-op-level TM overhead (e.g. inside the
`_apply_full_rope` chain), but the Python-level audit shows nothing
removable. Closing this task as "AUDIT DONE, NO ACTION".

### Cumulative outcome

- **#191**: BLOCKED (contract violation). Reverted; effort to actually
  land is 2-4h. Promoted to **NEXT-F**.
- **#192**: AUDIT DONE, no action. Closed.

**Recommended next step**: send the email reply (§6) but downgrade the
"#44946 question" — instead of "is the fused op in main?" (we now
know it is), ask the more useful question: "what's your preferred
disjoint-shard pattern for NKV_PER_CHIP=1 / per-KV-head split layouts?
The Llama70b shape-restricted RM path doesn't fit, and we'd rather
fork your pattern than invent a new one."

------------------------------------------------------------------------

## 3. NEXT batch — to schedule after Nemotron G4 lands

Each gets its own plan-of-action doc (TBD). Sketch only here:

### NEXT-A: GDN kernel bake-off (Tracy A/B)

- Install `recurrent_gated_delta_rule_decode_ttnn` from the
  `changh95/qwen3-coder-next-wh-qb` branch on a side build.
- Run our 35B traced decode with both kernels at fixed (B, num_heads,
  state_shape).
- Compare per-token kernel time (Tracy), correctness PCC vs our
  validated reference.
- Decide: keep ours, swap to theirs, or upstream a merged design.
- Deliverable: `research/gdn_kernel_bakeoff_plan.md` + Tracy delta table.

### NEXT-B: RMSNorm fusion on Gemma 4

- Read tt-metal #44948 implementation details.
- Decide between: (a) distributed-RMSNorm sharded across cores
  (matches our Gemma 4 perf briefing P2), or (b) fused-into-following-
  matmul pattern.
- Probe via `experiments/cb/isolate/gm4_rmsnorm_*.py` (new); compare
  fused vs current.
- Projected win: +12-15 ms/tok per the briefing.
- Deliverable: kernel/wrapper change + Tracy bench + correctness gate.

### NEXT-C: Chunk-outer 2048-tok prefill trace

- Reframe our `forward_prefill_chunked_tp` from "1 tok/iter for
  L > chunk_size=32" to "capture one chunk's all-layer forward,
  replay per chunk to 128K".
- Massive cold-TTFT lever (per Qwen audit §3).
- Deliverable: new `forward_prefill_chunked_2048_traced` + CB
  integration + bench.

### NEXT-D: Masked fixed-bucket prefill

- New isolation probe + integration of {128, 256, 512, 1024, 2048}
  bucketing.
- Smaller program-cache footprint than the chunked variant.
- Mostly perf engineering; depends on NEXT-C scaffolding.

### NEXT-E: Wider `deltanet_recurrence` op signature

- Redesign our `qwen36_decay_gate_decode_owned` to absorb additional
  projections.
- Substantial owned-ops work — equivalent to a G0..G4 ladder.

------------------------------------------------------------------------

## 4. Communications (item 10)

Three email replies are drafted verbatim in §6 of each audit doc.
Send each independently:
- gm4: cite specific commits we've shipped + ask about `paged_fused_
  update_cache` signature + RMSNorm fusion target pattern.
- qwen: ask about the "256K vs 128K" discrepancy + offer our PC +
  active-prompt suffix detector.
- gdn: send our complete-kernel pointer + ask about TT-LANG ablation.

User to send when convenient (no automation, no PRs without explicit
go-ahead).

------------------------------------------------------------------------

## 5. Tasks (tracked in the project task list)

- **#191** — adoption NOW-1: `paged_fused_update_cache` on Gemma 4
- **#192** — adoption NOW-2: redundant `to_memory_config` audit on Gemma 4
- **#193** — adoption NEXT-A: GDN bake-off plan (post-G4)
- **#194** — adoption NEXT-B: RMSNorm fusion on Gemma 4 (post-G4)
- **#195** — adoption NEXT-C: chunk-outer 2048-tok prefill trace (post-G4)
- **#196** — adoption NEXT-D: masked fixed-bucket prefill (post-NEXT-C)
- **#197** — adoption NEXT-E: wider deltanet_recurrence op signature (post-G4)

------------------------------------------------------------------------

## 6. Risk register

| Risk | Mitigation |
|---|---|
| Adoption subagent's edits break Gemma 4 multi-turn correctness | Smoke gate (multi-turn HTTP) is mandatory before commit |
| Subagent and main agent both push concurrent commits to same files | Adoption touches `server_gemma4_unified_ttnn.py` only; main agent's Nemotron work touches `experiments/owned_ops/nemotron3_mamba2_decode_owned/` only. ZERO file overlap. |
| Tracy measurement noisy → false "no win" conclusion | Subagent must take 3 runs, report median + IQR |
| Server restart contention with main agent's qb1 work | Main agent's Mamba kernel work doesn't need the running CB server — only the kernel build + harness. Subagent can restart freely. |

------------------------------------------------------------------------

## Related

- `research/audit_gemma4_opts_us_vs_tt_metal_44962.md`
- `research/audit_qwen36_us_vs_qwen9b_p150_branch.md`
- `research/audit_gdn_kernel_us_vs_tt_metal.md`
- `research/nemotron3_nano_30b_a3b_bringup_plan.md` (foreground work)
- `research/mm7_g1_mamba2_kernel_design.md` (foreground work)
