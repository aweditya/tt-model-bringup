# Owned conv1d Investigation Log — 2026-05-18 evening → 2026-05-19 small hours

Current status: **owned_conv1d is NOT in production** and remains disabled
by default (`state.deltanet_conv1d_mode = "manual"`). owned_gdn ships
unchanged at 12.46 tok/s. Investigation suspended; resume next session.

## What works (PROVED)

| level | result | citation |
|---|---|---|
| Kernel correctness on synthetic single-device | PCC 0.99999 at D=2560 | `50a6555`, `owned_conv1d_g0_D2560_20260519_0310.json` |
| Single-step slice + kernel + writer on single-device | All 4 checks pass at BF16-ULP scale | `e168c4d` (CHECK 1-3), `20f8d69` (CHECK 4) — `owned_conv1d_state_shift_check_20260519_0655.json` |
| Production decode with `--deltanet-conv1d-mode manual` (toggle exists) | 80.43-80.49 ms/tok unchanged | multiple control runs this session |

## What fails (REPRODUCED)

| variant | top-1 disagreement | first flip | min cosine | commit |
|---|---|---|---|---|
| G3: per-step slice/concat/copy_back wire-in | **39/500 = 7.8%** | step 17 | 0.7346 | `df1cccc` |
| G4: bootstrap pre-split tensors | **66/500 = 13.2%** (WORSE) | step 0 | 0.0167 | `142d63e` (still in-tree, gated behind flag) |

Both failures: large immediate divergence, NOT slow drift. Both reproduce
through the resident server's mesh-sharded `deltanet_step_tp` owned branch.

## What we ruled out

1. **Kernel itself is buggy** — refuted by G0 + G1 standalone passes.
2. **`ttnn.slice` puts data at wrong tile column** — refuted by G1
   CHECK 1 (slice readback matches numpy at BF16-ULP scale).
3. **Writer kernel mis-shifts state** — refuted by G1 CHECK 4 (state
   shift `s0=state1, s1=state2, s2=mixed` all correct at BF16-ULP).
4. **Per-step slice/concat/copy_back chain at single-device** — refuted
   by G1 single-device single-step pass.

## What remains untested (next-session candidates)

1. **Mesh-aware kernel dispatch** — all G0/G1 probes ran single-device.
   The kernel might behave differently when called on mesh-sharded
   tensors via TTNN's auto-dispatch. Owned_gdn works on mesh, so it's
   POSSIBLE but the conv1d kernel may have a subtle mesh-incompatible
   pattern.
2. **owned_conv1d × manual_gdn interaction** — the isolation test
   (`--modes manual --deltanet-conv1d-mode owned_conv1d`) hung because
   of the cumulative-slowdown issue (see below); we never got a clean
   answer on whether owned_conv1d works without owned_gdn active.
3. **Pre-split data correctness on mesh** —
   `dn['w_conv_split'][k] != dn['w_conv'][:, k:k+1]` is possible if
   `upload_sharded(relayout_conv(np[:, k:k+1]), dim=0)` doesn't produce
   the same per-chip distribution as
   `upload_sharded(relayout_conv(np), dim=0)[:, k:k+1]`. Single-device
   probe can't catch this — needs a mesh-aware tensor-equality probe.

## Newly-confirmed workflow gotcha (2026-05-19 small hours)

The "owned_gdn 2nd-invocation eager slowdown" documented in commit
`2905470` is **NOT specific to owned_gdn**. It applies to ANY custom
owned-mode kernel invocation in eager mode. Specifically:

- Multiple cosine_ladder_tp calls in the same server lifetime work
  fine IF the modes don't keep toggling (e.g., 3 calls with the same
  modes worked in the 07:08-07:14 sequence).
- A cosine_ladder_tp call that TOGGLES `state.deltanet_recurrence_mode`
  (e.g., from default `owned_gdn` to `manual`) seems to work the FIRST
  time but HANG on the SECOND such-toggling call (e.g., 07:43 manual
  control worked, 07:44 manual+owned_conv1d hung indefinitely at
  100%+ CPU for 30+ min).

Working hypothesis: JIT cache or L1 fragmentation in the
mode-toggling path. The Python-level branching in `deltanet_step_tp`
between owned/manual paths may compile differently on each first-touch,
and the second-touch hits some state where one of the JIT'd kernels
deadlocks or thrashes.

This makes diagnostic probing expensive — every test variant requires
a fresh ~17-min bootstrap. Future custom-op bring-ups should plan for
"one mode test per server lifetime" until this is root-caused.

## Recommended next-session work

### Step 1 — write a mesh-aware single-forward comparison probe (~1 hr)

Server endpoint `handle_probe_deltanet_owned_conv1d_real_mesh_tp` that:

- Takes a layer index (default 0)
- For ONE forward call:
  - Reads `dn['conv_st']` and `dn['w_conv']` (combined)
  - Reads `dn['conv_st_split'][k]` and `dn['w_conv_split'][k]` for k in range
  - **VERIFIES** `to_torch(split[k]) == to_torch(combined)[:, k:k+1]` per chip
  - Synthesizes mixed_qkv
  - Runs manual conv1d block → conv_out_manual
  - Runs owned conv1d block → conv_out_owned
  - Compares conv_out element-wise + state-shift result
- Returns comparison stats + the tensor-equality answers

This single probe localizes whether the bug is:
- (a) `dn['w_conv_split']` doesn't equal slices of `dn['w_conv']` →
  pre-split upload bug on mesh
- (b) Pre-split tensors are equivalent but the kernel produces different
  output on mesh → mesh-aware kernel-dispatch bug
- (c) Both layers correct but state-shift over multiple steps diverges
  → multi-step state-write bug (would need multi-step probe extension)

### Step 2 — fix based on what step 1 finds

Most likely fix paths in priority order:
- If (a): fix `relayout_conv` to handle single-column input correctly,
  OR pre-split at the relayout level (split first, relayout each
  column independently — likely equivalent but safer).
- If (b): the kernel's program-factory may need explicit mesh handling.
  Look at how owned_gdn (which works) registers itself for mesh
  dispatch and mirror.
- If (c): explicit mesh-aware writer kernel that handles per-chip
  state buffer addresses correctly.

### Step 3 — re-run G3 after fix; ship if ≤2% disagreement

## Pivot recommendation for the rest of this session arc

owned_gdn is unaffected and shipping at 12.46 tok/s. The next-best work
items (per the post-owned-gdn fusion roadmap) are independent of
owned_conv1d:

1. **decay/gate G0 test on qb2** — already built + .so synced (commit
   `db7531e`). Should take 1 fresh bootstrap + 1 test command to
   validate. If G0 passes, decay/gate gets the same G1/G2/G3 sequence.
2. **Native partial RoPE landing** — plan in
   `research/native_rope_landing_plan_2026_05_18.md`. No new kernel
   needed; just G3 + G4. Cheapest projected per-day win.

owned_conv1d returns when we have fresh eyes + the mesh-aware probe
implemented.

## Artifacts produced in this investigation

```text
.cache/qb2_tp_deltanet/owned_conv1d_g0_D{32,128,2560}_20260519_0310.json
.cache/qb2_tp_deltanet/owned_conv1d_slice_hypothesis_20260519_0650.json
.cache/qb2_tp_deltanet/owned_conv1d_state_shift_check_20260519_0655.json
.cache/qb2_tp_deltanet/cosine_ladder_tp_compare_conv1d_500_20260519.json       (G3 FAIL: 7.8%)
.cache/qb2_tp_deltanet/cosine_ladder_tp_compare_conv1d_g4_500_20260519.json    (G4 FAIL: 13.2%)
```

## Code state (no rollback required)

The G4 fix (commit `142d63e`) is still in-tree. Bootstrap pre-allocation
of `dn['conv_st_split']` and `dn['w_conv_split']` happens for every
DeltaNet layer regardless of mode. Memory cost ~1.7 MB/chip total —
negligible. When `state.deltanet_conv1d_mode == "manual"` (the default
in production), these tensors are unused but allocated.

To revert if memory becomes a concern: undo `142d63e` and the related
parts of `20f8d69`. Until then, no action needed.
