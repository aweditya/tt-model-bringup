# MoE Bringup Cleanup Plan (35B-A3B)

Senior-engineer PR-review notes for the MoE path. The user's complaint: code
is verbose, has multiple living iterations of the same kernel, redundant CLI
flags, and historical-narrative comments. Goal: make the MoE bringup
human-readable while preserving every load-bearing rationale (especially
multi-day debug findings).

## TL;DR

- The biggest win is **retiring `moe_forward_ttnn_pattern_a` (the looped
  variant)** + its CLI mode. `pattern_a_batched` strictly dominates it
  (146 vs 308 ms/tok, same algorithm). Keep `topk` as the A/B reference; kill
  the middle path. **~160 lines** disappear from `server_35b_ttnn.py`.
- **Move `test_batched_expert_matmul_isolated.py` to
  `experiments/utils/archive/` and replace it with a 1-variant regression
  test** (variant H — the production shape). The 12-variant matrix is
  *documentation* of dead ends, not a runtime suite.
- **Strip `_DBG` / `__debug_shapes` instrumentation** from
  `moe_forward_ttnn_pattern_a` (~30 lines, currently dead code in
  `pattern_a_batched` anyway).
- **Tighten ~25 comment blocks** that read as session diaries (`B17-B v3 …`,
  `Bug isolation: …`, `FIRST PASS: …`). Keep all `Why:`-style rationale on
  view-decay, `+1` RMSNorm, K-broadcast RoPE — they were load-bearing.
- Consolidate `upload_moe_layer` / `upload_moe_layer_pattern_a` and
  `moe_forward_ttnn_pattern_a_batched` / `moe_forward_ttnn` shared bottoms
  (router + shared-expert) into 2 small helpers.

## Per-file cleanup plan

### `experiments/serve/server_35b_ttnn.py` (1986 lines)

**Remove (~250 lines):**
- `moe_forward_ttnn_pattern_a` (lines 1161-1316, the looped variant). Strictly
  dominated by `_batched`; same upload format, same correctness. Keep the
  rationale in a 4-line block comment above `_batched` that says
  "looped variant retired 2026-05-25, see commit 961ce7f and
  research/35b_moe_pattern_a_plan.md".
- The `_DBG` / `__debug_shapes` blocks inside the (now-removed) looped
  variant — they were debug scaffolding for the view-decay hunt, captured in
  the isolated test suite already.
- Dispatch arm at L1042 (`state.moe_mode == "pattern_a"` branch) and L1878.
- `state.moe_mode` becomes a 2-valued field: `"topk"` (A/B reference) and
  `"pattern_a_batched"` (default after this cleanup). Default flip can be a
  separate commit.

**Keep (load-bearing, looks removable but isn't):**
- The `topk` path (`moe_forward_ttnn` + `upload_moe_layer`). It's the
  numpy-oracle-validated reference; we A/B every perf change against it. The
  host-readback `to_torch` in topk is the SAME pattern as
  `test_pattern_a_moe_np.moe_topk_np` — keep both for drift attribution.
- All `Why:` comments on `routing_weight_3d` (clone-after-view, L1356-1359),
  `h_3d` (don't-dealloc view, L1376-1379), and the
  `mul(expert_out, rw) + sum → matmul(rw, expert_out)` fusion (L1397-1406).
  These are the rationale that lost ~half a day to debug. They stay verbatim.
- The `cs_rank == 4` / `rank == 3` dual-paths in `dn_forward_ttnn` for
  shape-rank polymorphism on mesh-sharded vs replicated buffers — the rank
  discrepancy is a ttnn quirk, not historical baggage. Keep but compress the
  comment.

**Tighten (comment-shortening, no behavior change):**
- L2-24 module docstring: shrink from 23 lines to ~8. Drop the B16 phase
  history and "Companion to server_35b.py" reference; that's git/research
  territory.
- L82-86 `Pattern A` rationale: trim to one sentence pointing at
  `research/35b_moe_pattern_a_plan.md`.
- L93-105 HIFI4 block: 13 lines explaining "every linear gets HiFi4". One
  line: `# HiFi4 + fp32_dest_acc on every matmul — matches 27B 91f recipe.
  Why: fixes accumulated bf16 noise at L31/L39.`
- L595-617 RMSNormGated narrative ("Bug isolation: ttnn.rms_norm output
  cos 0.9582…") — the bug is fixed; collapse to one line:
  `# RMSNormGated: rms_norm WITHOUT weight, then explicit ttnn.mul (weight
  fused into rms_norm gave cos 0.9582 vs oracle).`
- L645-658 `B17-B-DN: in-place state update…` comment block (14 lines) →
  4 lines, drop the "Same pattern as 27B server_tp.py:938-944" reference
  (git history covers it).
- L902-908 `RoPE deferred…NOOP_ROPE=True` historical note. ENABLE_ROPE is
  always True now; delete the dead-flag narrative. Keep one line saying
  "K-broadcast workaround for single-row [1, HEAD_DIM] ttnn slice/concat
  bug, see feedback_qwen36_attn_rope_single_row_ttnn_bug.md".
- L1071-1083 `NOTE: ttnn.embedding…` and `B17-B v3 …` blocks in
  `moe_forward_ttnn` (topk): collapse to one line — the host-readback is
  what makes this path non-traceable. That's the only fact the comment
  needs to convey.
- Remove `ENABLE_ROPE`/`BROADCAST_KV` local flags in
  `attn_forward_ttnn_manual` (L909-910) — they are always True. Inline the
  branches. Keep the K-broadcast `Why:` comment.

**Code structure (small helpers):**
- Extract `_moe_router_topk(h_tt, w)` shared by both `moe_forward_ttnn`
  (topk) and `moe_forward_ttnn_pattern_a_batched`. Returns `(top_vals,
  top_idxs, weights_norm)`. 8 lines saved per call site × 2.
- Extract `_moe_shared_expert(h_tt, w, mesh)` — identical 11-line block in
  both forwards. Saves another ~20 lines.
- Extract `_moe_upload_shared(sd, mesh)` for the shared-expert weight upload
  (5 lines, duplicated in `upload_moe_layer` + `upload_moe_layer_pattern_a`).
  Saves ~10 lines and reads better.

### `experiments/utils/test_pattern_a_moe_np.py` (188 lines)

**Keep as-is, but:**
- Tighten the 13-line docstring at the top to ~5 lines (state goal + assert
  thresholds, drop the bullet list). The file is already a clean reference
  implementation.

### `experiments/utils/test_pattern_a_moe_tt.py` (104 lines)

**Tighten:**
- L14-21 inline-env-export run block: replace with a one-liner reference to a
  helper script (or to the canonical "how to run stuff" memory). Don't paste
  9 lines of env vars in every test docstring.
- Remove `debug_capture = {"__debug_shapes": True}` plumbing (L68) once the
  `_DBG` blocks in `moe_forward_ttnn_pattern_a` are removed.

### `experiments/utils/test_batched_expert_matmul_isolated.py` (680 lines)

**The biggest single cleanup.** This file is a *log* of 12 variant attempts;
6 fail, 6 pass; only H is in production. Per the user's "iterations belong
in a separate folder" directive:

**Move** the whole file to `experiments/utils/archive/test_batched_expert_matmul_variants_2026_05_25.py`
verbatim — preserving the debugging log for future ttnn-shape-debug sessions.

**Replace** the original path with a ~120-line `test_batched_expert_matmul_isolated.py`
that:
- Keeps the synthetic-setup boilerplate (h_np, weights_per_chip_np,
  reference_loop_np).
- Keeps **only variant H** (rank-3 [256,H,2I] sharded W + on-device
  `ttnn.repeat` for h) as `test_production_shape()`, the regression gate.
- Adds a one-line summary table in the docstring pointing at the archived
  file for the historical context (A failed because rank-4 broadcast; B
  failed because expert-sharded; etc).

**Why this matters:** the production shape can change again (e.g. if we
switch to E_LOCAL=32 for a different model). The 6 failing variants are
debugging history, not regression tests. Keeping them in the live test bloats
review and forces every reader to triage which variants matter.

### `experiments/utils/moe_router_compare_35b.py` (126 lines)

**Keep as-is.** This is the H3 probe (router stability) — its hypothesis got
rejected but the probe is a clean template for any future TT-vs-HF
top-k-disagreement investigation. ~120 lines, no historical narrative, no
flag soup. Already in good shape.

**Tighten only:**
- The 33-line docstring is long but every paragraph is "what does this
  measure". Trim the duplicated reference between L4-14 (TT/HF softmax
  precision) and the body — 5 lines saved.

### `experiments/utils/profile_blocks_35b_ttnn.py` (159 lines)

**Keep as-is.** Tight monkey-patch, sync-bounded, one purpose. The 8-line
docstring is exactly the right length.

## Folder reorganization

Adopt **`experiments/utils/archive/`** for retired-but-historically-valuable
files. Convention:
- File still runs (so it documents itself); not imported by anything live.
- Top of file: 5-line note saying "archived YYYY-MM-DD, see
  research/<doc>.md for the lesson, see commit <sha> for context".
- No CI / no scheduled runs.

Initial contents proposed by this cleanup:
- `archive/test_batched_expert_matmul_variants_2026_05_25.py` (the 12-variant
  matrix).

Do **not** create `experiments/utils/iterations/` — too generic. `archive/`
is unambiguous.

## CLI / API surface trim

| Flag | Where | Commit | Verdict |
|---|---|---|---|
| `--moe-mode pattern_a` | `profile_35b_ttnn.py`, `trace_demo_full_step.py` | `5f4cff8` (introduced 2026-05-25) | **REMOVE** — superseded by `pattern_a_batched` (same upload, strictly faster). |
| `--moe-mode topk` | `profile_35b_ttnn.py` | `5f4cff8` | **KEEP** — A/B reference for any future MoE perf claim. |
| `--moe-mode pattern_a_batched` | `profile_35b_ttnn.py`, `trace_demo_full_step.py` | `961ce7f` (2026-05-25) | **KEEP, make default** in `state.moe_mode`. |
| `__debug_shapes` (dict key) | `test_pattern_a_moe_tt.py`, `_DBG` blocks in server | – | **REMOVE** — was for the routing_weight view-decay hunt; bug fixed in `961ce7f`. |
| `sub_capture` (dict) | `dn_forward_ttnn`, `attn_forward_ttnn`, `moe_forward_ttnn*`, `layer_forward_ttnn` | – | **KEEP** — actively used by `cosine_ladder_35b.py`, `hf_reference_35b.py`, `layer_subop_cos_35b.py` for drift attribution. The 93 `sub_capture` usages are not bloat; they are the diagnostic backbone. |

## Risk register

1. **Retiring `pattern_a` (looped) loses the A/B reference against
   `pattern_a_batched`.** Mitigation: `topk` is the better A/B target anyway
   (different algorithm, different upload, different host-readback profile).
   `pattern_a` was only a stepping stone. Before deleting, run
   `test_pattern_a_moe_tt.py` one final time as topk-vs-batched and capture
   the cos in `research/archive/35b_moe_pattern_a_batched_status.md`.

2. **Helper extraction (`_moe_router_topk`, `_moe_shared_expert`)
   risks subtle ordering changes** (e.g. dealloc timing of `h_tt`). Mitigation:
   commit the extraction separately, run `bench_decode` before/after, and
   diff against the 146 ms/tok number from
   `research/35b_perf_milestones.md`. Anything outside ±2% reverts.

3. **Moving the variant suite to `archive/` loses the regression coverage
   of the 5 passing variants we don't ship (D, F, G, I, J, K, N, O).** Risk
   is low: those passes were "this ttnn-shape combo also works" data points,
   not behavioral guarantees of the model. The keepers (H, batched-matmul
   production shape + view-decay-survives-FFN-chain) become the live test.

4. **Comment-tightening risks losing rationale that future-us needs.**
   Mitigation: do this commit-by-commit, smallest unit each. For every
   removed paragraph, ask: "is the WHY captured elsewhere (memory file,
   research doc, the `Why:` line on the surviving comment)?" If no, keep.

5. **Default-flip `state.moe_mode = "pattern_a_batched"`** changes behavior
   for any existing tooling that didn't pass `--moe-mode`. Mitigation: do
   the default flip last, after every caller is verified to either set the
   mode explicitly or be on the new default. Audit callers:
   `profile_blocks_35b_ttnn.py` (no flag — needs update),
   `test_pattern_a_moe_tt.py` (sets explicitly — safe), all `tracy_*` probes
   (set explicitly — safe).

## Execution order

Each step is a separate commit, independently testable.

1. **Commit A — "35B MoE: remove `_DBG` / `__debug_shapes` instrumentation"**
   Pure deletion. Run `test_pattern_a_moe_tt.py` (topk vs pattern_a still
   exists at this point) to confirm cos unchanged.

2. **Commit B — "35B MoE: retire looped Pattern A in favor of batched"**
   Delete `moe_forward_ttnn_pattern_a`, `state.moe_mode == "pattern_a"`
   arms, the `--moe-mode pattern_a` choice in
   `profile_35b_ttnn.py` / `trace_demo_full_step.py`. Update
   `test_pattern_a_moe_tt.py` to compare topk vs `pattern_a_batched`.
   Verify: `bench_decode` perf unchanged on `pattern_a_batched`.

3. **Commit C — "35B MoE: archive variant matrix, keep production shape
   regression"** Move the 680-line variant test to
   `experiments/utils/archive/`, write a new ~120-line
   `test_batched_expert_matmul_isolated.py` with just variant H + the
   shared setup. Run it; should PASS in ≤10 s.

4. **Commit D — "35B MoE: extract router + shared-expert helpers"**
   Pull `_moe_router_topk` and `_moe_shared_expert` out of
   `moe_forward_ttnn` and `moe_forward_ttnn_pattern_a_batched`. Pull
   `_moe_upload_shared` out of `upload_moe_layer` and
   `upload_moe_layer_pattern_a`. Run `bench_decode`; perf must be within
   ±2% of 146 ms/tok.

5. **Commit E — "35B MoE: tighten comments and module docstrings"**
   The comment-shortening pass on `server_35b_ttnn.py`. Strictly
   non-functional; visual diff review only. Preserves every `Why:` line.

6. **Commit F (optional, behavior-change) — "35B MoE: default
   `state.moe_mode = 'pattern_a_batched'`"** Last; after every caller is
   audited. Smoke-test the server end-to-end.

After all six commits the MoE path is ~400 lines shorter end-to-end, with
the same perf profile and the same correctness gates. Every multi-day debug
finding is either still inline as a one-line `Why:` or referenced in
`research/` / memory files.
