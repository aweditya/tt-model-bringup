# Integration outline: vocab-sharded LM head (candidate #6, DeepSeek pattern)

> Status: design only — local-research output. No device runs. Ship after maintainer
> review + a single qb2 validation experiment (described below).

## Summary

- **Pre-state (today, `server_tp.py:277, 673, 837`):** lm_head is uploaded
  `ReplicateTensorToMesh` as `[HIDDEN, VOCAB] = [5120, 152064]` bf16
  (~1.56 GB / chip). Every chip executes the full matmul, then we
  `ConcatMeshToTensor(dim=0)`-read all 4 replicas and take row 0.
  - P3 probe measured **4.16 ms median matmul** at 372 GB/s = 72% of 512 GB/s
    P150 peak (`feedback_p3_lm_head_replicated_pass.md:15-18`).
  - Plus a `ttnn.to_torch` of `[1, 152064]` bf16 = ~304 KB PCIe per token.
    Memory-index estimate of 9.4 ms readback is consistent with prior
    `performance_ceilings.md:60` projections at this vocab size on mesh
    (single-chip P150 was 0.3 ms for ~half the vocab) — call it
    **~5–9 ms readback** today and validate with Tracy if challenged.
- **Post-state (DeepSeek pattern):**
  - Shard lm_head along the vocab axis: per-chip slab
    `[HIDDEN, VOCAB/4] = [5120, 38016]` (388 MB / chip).
  - Per-chip linear produces `[1, 38016]`.
  - On-device per-chip `(argmax, max_value)` → 4 (idx, val) pairs total.
  - Host reads 4 ints + 4 floats (≈48 bytes), picks the chip with the
    largest value, returns `chip_idx * 38016 + local_idx`.
- **Estimated wins:**
  - Matmul: ~4× less work per chip → ~**1.0–1.5 ms** (consistent with
    `feedback_p3_lm_head_replicated_pass.md:21-25` "compute scales ~4×:
    4.16 → ~1.05 ms").
  - Readback: 152064 floats → 8 numbers → effectively eliminates the
    PCIe leg (sub-ms).
  - Together: **~8–12 ms/tok saved**, consistent with the v2 addendum
    upward revision (`reference_multi_chip_opt_menu_v2.md:17`).
- **Risk:** low for greedy decode (current path).
  - We already have argmax mismatch only on razor-thin HF top-1 margins
    (`feedback_long_context_cosine_ladder.md` cosine ladder 97/100 match).
    With shard cos 0.999919 on the matmul side
    (`feedback_p3_lm_head_replicated_pass.md:13`), per-chip-then-global
    argmax produces the same id whenever bf16 currently does.
- **Out of scope (defer):** full-distribution sampling (temperature/top-p/DRY,
  `feedback_drift_dry_rep_penalty.md`) needs the full `[1, 152064]` logits
  vector. Path: fall back to the old `all_gather + ConcatMeshToTensor`
  whenever `temperature > 0` (or other sampler kwargs are present). The
  client today only sends greedy, so the fast path is the default.

---

## Code changes (server_tp.py)

> Citations: line numbers are from the file at the time of writing
> (`/Users/adityasriram/Labs/stanford/cs440lx/tt-xla/experiments/serve/server_tp.py`,
> 982 lines).

### Change 1 — Bootstrap: shard upload (around line 277)

**Before** (`server_tp.py:267-278`):

```python
# === Stage B (cont): embed, lm_head, final_norm — replicated ===
print(f"[bootstrap] loading embed + lm_head + final_norm + RoPE tables…", flush=True)
# Reuse 91l's loader (used by single-chip server too)
spec2 = importlib.util.spec_from_file_location(
    "_91l", os.path.join(PROJECT_ROOT, "experiments", "91l_fp32_residual_generate.py"))
_91l = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(_91l)
embed_weights = _91l.load_embed_lm_head_weights()
state.embed_np = embed_weights['embed']
state.final_norm_tt = upload_replicated(embed_weights['final_norm'])
state.lm_head_tt = upload_replicated(embed_weights['lm_head'])
print(f"  ✓ embed/lm_head/final_norm uploaded", flush=True)
```

**After:**

```python
# === Stage B (cont): embed, lm_head, final_norm ===
print(f"[bootstrap] loading embed + lm_head + final_norm + RoPE tables…", flush=True)
# Reuse 91l's loader (used by single-chip server too)
spec2 = importlib.util.spec_from_file_location(
    "_91l", os.path.join(PROJECT_ROOT, "experiments", "91l_fp32_residual_generate.py"))
_91l = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(_91l)
embed_weights = _91l.load_embed_lm_head_weights()
state.embed_np = embed_weights['embed']
state.final_norm_tt = upload_replicated(embed_weights['final_norm'])
# DeepSeek `lm_head1d.py` pattern (`models/demos/deepseek_v3/tt/lm_head1d.py:67-82`):
# shard lm_head along the vocab axis. Our weight is stored [HIDDEN, VOCAB] (see
# `91l_fp32_residual_generate.py:82` — the .T) so vocab is dim=1.
# Per-chip slab is [HIDDEN, VOCAB/NCHIPS] = [5120, 38016].
NCHIPS = state.mesh.get_num_devices()
VOCAB = embed_weights['lm_head'].shape[1]
assert VOCAB % NCHIPS == 0, f"vocab {VOCAB} not divisible by nchips {NCHIPS}"
state.vocab_per_chip = VOCAB // NCHIPS  # 38016 for 4 chips
assert state.vocab_per_chip % 32 == 0, \
    f"per-chip vocab {state.vocab_per_chip} not tile-aligned"
state.lm_head_tt = upload_sharded(embed_weights['lm_head'], dim=1)
state.vocab_size = VOCAB
print(f"  ✓ embed/lm_head/final_norm uploaded "
      f"(lm_head sharded dim=1, per-chip {state.vocab_per_chip})", flush=True)
```

Also add the `vocab_size` and `vocab_per_chip` fields to `MeshServerState.__init__`
(`server_tp.py:65-96`) for cleanliness:

```python
        self.vocab_size = None       # full vocab (152064)
        self.vocab_per_chip = None   # VOCAB // NCHIPS (38016 at 4 chips)
```

### Change 2 — Forward: per-chip linear (`server_tp.py:651-674`)

`forward_token_tp_inner` returns `logits_tt` which is consumed both by the
captured trace and by the decode loop reader. After Change 1, the same
`ttnn.linear` call now produces `[1, VOCAB/NCHIPS]` on each chip — no edit
needed in the linear call itself.

**Before** (`server_tp.py:672-674`):

```python
x_tt = _rms_norm_manual(x_tt, state.final_norm_tt, 1e-6, HIDDEN)
logits_tt = ttnn.linear(x_tt, state.lm_head_tt)
return logits_tt
```

**After (same linear; just add the on-device reduction so the trace returns
small tensors):**

```python
x_tt = _rms_norm_manual(x_tt, state.final_norm_tt, 1e-6, HIDDEN)
# Per-chip logits [1, VOCAB/NCHIPS]. Sharded-output: each chip holds its
# own vocab slab. DeepSeek skips the final all_gather entirely
# (`lm_head1d.py:242-251` — `forward_decode` returns the sharded output).
per_chip_logits = ttnn.linear(x_tt, state.lm_head_tt)
# On-device per-chip argmax (idx) + max (value). Galaxy uses
# `ttnn.argmax(..., dim=3, keepdim=True, use_multicore=True)` after
# all-gathering (`llama_model.py:607`, `demo_performance.py:244`). We skip
# the gather: each chip argmaxes its OWN slab and we resolve on host.
per_chip_argmax = ttnn.argmax(per_chip_logits, dim=-1, keepdim=True)  # [1, 1]
per_chip_max    = ttnn.max(per_chip_logits, dim=-1, keepdim=True)     # [1, 1]
return per_chip_logits, per_chip_argmax, per_chip_max
```

The traced graph now yields a 3-tuple. Touchpoints to update:

- `state.traced_logits_tt` (single tensor) → `state.traced_outputs` (3-tuple).
  Adjust `MeshServerState.__init__` (`server_tp.py:81`) accordingly.
- `_ensure_decode_trace` (`server_tp.py:738-768`): the captured-region call
  now needs to capture the tuple. `ttnn.begin_trace_capture` is agnostic;
  storing the tuple in `state.traced_outputs` is enough.

```python
def _ensure_decode_trace(state):
    ...
    update_input_buffers(state, token_id=0, cur_pos=2)
    state.trace_id = ttnn.begin_trace_capture(state.mesh, cq_id=0)
    state.traced_outputs = forward_token_tp_inner(state)   # (logits, argmax, max)
    ttnn.end_trace_capture(state.mesh, state.trace_id, cq_id=0)
```

- `_traced_forward` (`server_tp.py:771-776`) returns the tuple too:

```python
def _traced_forward(state, token_id, cur_pos):
    import ttnn
    update_input_buffers(state, token_id, cur_pos)
    ttnn.execute_trace(state.mesh, state.trace_id, cq_id=0, blocking=False)
    return state.traced_outputs   # (per_chip_logits, per_chip_argmax, per_chip_max)
```

### Change 3 — Output read: 8-number readback + host resolve (`server_tp.py:835-839`)

**Before:**

```python
for step in range(max_tokens):
    # Read logits from chip 0 (mesh-composed → first row is chip 0's view)
    logits_t = ttnn.to_torch(last_logits, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0))
    logits_np = logits_t.float().cpu().numpy().reshape(-1)[: state.embed_np.shape[0]]
    next_id = int(np.argmax(logits_np))
```

**After:**

```python
for step in range(max_tokens):
    per_chip_logits, per_chip_argmax, per_chip_max = last_logits
    next_id = _resolve_sharded_argmax(state, per_chip_argmax, per_chip_max)
```

Add the helper near `_traced_forward` (one place; ~15 LOC):

```python
def _resolve_sharded_argmax(state, argmax_tt, max_tt):
    """Resolve global argmax from per-chip (local_idx, local_max) pairs.

    argmax_tt and max_tt are sharded on the mesh: each chip holds [1, 1]
    with its OWN local-slab argmax/max. We compose via
    `ConcatMeshToTensor(dim=0)` to get [NCHIPS, 1] for each. Host picks
    the chip whose max value is largest, then returns
    chip_idx * vocab_per_chip + local_argmax.

    Total readback: NCHIPS * 8 bytes (4 ints + 4 floats) per token.
    """
    import ttnn, numpy as np
    idx_concat = ttnn.to_torch(argmax_tt,
        mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0))
    val_concat = ttnn.to_torch(max_tt,
        mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0))
    # idx_concat shape: [NCHIPS, 1]; val_concat shape: [NCHIPS, 1]
    idx_np = idx_concat.cpu().numpy().reshape(-1).astype(np.int64)
    val_np = val_concat.float().cpu().numpy().reshape(-1)
    winner_chip = int(np.argmax(val_np))
    local_idx = int(idx_np[winner_chip])
    return winner_chip * state.vocab_per_chip + local_idx
```

### Change 4 — Trace capture verification (no edit; just verify)

Confirm during the first qb2 dry-run that:

1. `ttnn.argmax(per_chip_logits, dim=-1, keepdim=True)` is captureable inside
   `begin_trace_capture/end_trace_capture` on a (1,4) mesh.
2. `ttnn.max(..., dim=-1, keepdim=True)` same.
3. `_ensure_decode_trace` still completes (no JIT-during-capture hang —
   the warmup loop already handles this per
   `feedback_c4v4_validated.md`).

Galaxy `demo_performance.py:266-280` captures `argmax` *inside* the trace
region for line_all_gather'd logits, so the op is trace-friendly in
production. The risk is only whether our particular sharded layout +
multicore config plays nicely on Blackhole; if not, fall back to capturing
just the per-chip linear and running the reductions outside the trace
(extra ~0.1 ms each, still wins).

---

## DeepSeek `lm_head1d.py` recipe (canonical reference)

Weight conversion (`models/demos/deepseek_v3/tt/lm_head1d.py:70-82`):

```python
return {
    "linear": {
        "input_tensor_b": shard_and_save(
            output_path / "linear.input_tensor_b",
            weight_tensor,                       # [1, 1, vocab, hidden]
            shard_dims=(None, -2),               # shard along vocab (dim=-2)
            mesh_device=mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    }
}
```

Decode config validates divisibility + tile alignment
(`lm_head1d.py:93-119`):

```python
hidden_dim, vocab_size = cls._get_model_dims_from_cfg(hf_config)
tile_size = 32
mesh_cols = mesh_device.shape[1]
if vocab_size % mesh_cols != 0:
    raise ValueError(...)
if hidden_dim % tile_size != 0:
    raise ValueError(...)
n_per_device = vocab_size // mesh_cols
if n_per_device % tile_size != 0:
    raise ValueError(...)
```

Forward (`lm_head1d.py:242-251`):

```python
@classmethod
def forward_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig) -> ttnn.Tensor:
    assert x.memory_config() == cfg["input_memory_config"], ...
    output = cls._fwd_linear(x, cfg)   # ttnn.linear(x, **cfg["linear"])
    ttnn.deallocate(x)
    assert output.memory_config() == cfg["output_memory_config"]
    return output                       # SHARDED — no final all_gather
```

Note DeepSeek defines `AllGatherAsyncConfig` (`lm_head1d.py:160-165`) but
the `forward_decode` *does not call it*. The caller is responsible for
either gathering or doing the on-device argmax. That's the win.

---

## Sharding math (sanity check)

- Vocab `V = 152064`. NCHIPS = 4. Per-chip slab = `V / NCHIPS = 38016`.
- Tile alignment: `38016 / 32 = 1188` — clean (no padding needed).
- Hidden = 5120. `5120 / 32 = 160` — clean (already required).
- Weight `[HIDDEN, VOCAB]`. `ShardTensorToMesh(mesh, dim=1)` →
  per-chip `[5120, 38016]`. (Differs from DeepSeek's `[1,1,vocab,hidden]`
  + `shard_dims=(None, -2)` — same semantic since vocab is the sharded
  axis in both. We don't use `transpose_b` because our weight is already
  pre-transposed in `91l_fp32_residual_generate.py:82`.)
- Input `x_tt` is replicated `[1, 5120]` → `ttnn.linear` produces
  per-chip `[1, 38016]`.

---

## On-device argmax outline

| Step | Op | Shape per-chip | Mesh layout |
|---|---|---|---|
| 1 | `ttnn.linear(x, lm_head_sh)` | `[1, 38016]` | sharded on dim=-1 |
| 2 | `ttnn.argmax(...,dim=-1,keepdim=True)` | `[1, 1]` int32 | sharded |
| 3 | `ttnn.max(...,dim=-1,keepdim=True)` | `[1, 1]` bf16 | sharded |
| 4 | `to_torch(... ConcatMeshToTensor(dim=0))` | `[NCHIPS, 1]` | host |
| 5 | host: pick `winner = argmax(vals)`, `gid = winner * 38016 + idxs[winner]` | scalar | — |

Total readback: `2 × NCHIPS × tile-padded[1,1]`. Tensors come back tiled
so the host actually sees a small bf16 tile + int32 tile per chip, but
post-`to_torch` reshape to `[NCHIPS, 1]` discards the padding. PCIe
volume is dominated by the per-tile minimum (~32 elements each) — still
< 1 KB total vs 304 KB today. ttnn.argmax already validated to work on
single-chip in `experiments/55_readback_optimization.py:32-65`.

---

## Validation plan (single qb2 experiment, ≤ 5 min)

Write `experiments/utils/p_lm_head_sharded_argmax.py` (new probe; ~80 LOC):

1. Open (1,4) mesh + FABRIC_1D (mirror `p3_mesh_lm_head_replicated.py:60-90`).
2. Upload the *real* Qwen3.6-27B lm_head weight (use
   `load_embed_lm_head_weights()` from `91l_fp32_residual_generate.py:62`).
3. Build a numpy gold `x_np ~ N(0,1) [1, 5120]`, compute
   `gold = x_np @ lm_head_np`, record `gold_argmax = int(gold.argmax())`.
4. Mesh-sharded path: upload `x_replicated`, run the new `linear + argmax + max`
   chain, resolve via the helper above, get `tt_argmax`.
5. Assert `tt_argmax == gold_argmax`. Repeat for 5 different random `x`.
6. Time `linear + argmax + max + readback` for 30 iters; median latency.

Gate: all 5 prompts must match exactly + latency < 4.16 ms (today's matmul
alone). If both pass, ship.

(Following CLAUDE.md non-negotiable #6 — permanent file, no inline scripts.)

---

## Failure modes & fallbacks

1. **`ttnn.argmax` on mesh-sharded tensor unsupported.**
   Fallback: keep argmax+max OUTSIDE the captured trace — capture just
   `per_chip_logits`, then run reductions eagerly. Costs an extra dispatch
   pair per token (~0.2 ms) but preserves the bigger win.

2. **`ttnn.max(per_chip_logits, dim=-1, keepdim=True)` doesn't bf16-cleanly
   carry the bf16 logit value.** Cosine on 38016 entries should be fine
   — bf16 max of bf16 is exact. But if precision matters for tie-breaking,
   cast to fp32 first: `ttnn.typecast(per_chip_logits, ttnn.float32)` →
   then reduce. The argmax-of-bf16 is what HF effectively does today via
   our bf16 logits, so no regression.

3. **Sampler path (`temperature > 0`).** The fast path doesn't help —
   need full distribution. Add a gate in `handle_generate_tp`:

   ```python
   greedy = args.get("temperature", 0.0) == 0.0 and not args.get("dry_multiplier")
   if greedy:
       next_id = _resolve_sharded_argmax(state, per_chip_argmax, per_chip_max)
   else:
       # Fall back to all-gather + host sampling
       full_logits = _gather_full_logits(state, per_chip_logits)   # writes one
       next_id = _sample(full_logits, args)
   ```

   `_gather_full_logits` does `ttnn.all_gather(per_chip_logits, dim=-1)`
   then `ttnn.to_torch(..., ConcatMeshToTensor(dim=0))[0]` for chip 0's
   view. Implement only when sampler path is exercised; greedy is the
   prod path (`feedback_drift_dry_rep_penalty.md` notes DRY/rep-penalty
   work AT temperature=0, on the client side over the full vocab — but
   that sampling is currently client-side after a full-vocab readback
   anyway; either keep the gather-then-sample fallback OR move sampling
   server-side after this lands).

4. **`ConcatMeshToTensor(dim=0)` on a 1D-ish tile-padded tensor.**
   `[NCHIPS, 1]` requires the inner dim to be 1 element of a 32-element
   tile. `to_torch` strips the padding correctly (verified by
   `p3_mesh_lm_head_replicated.py:115-121` for the larger case).

5. **`shape[1] != mesh.shape[1]` mismatch.** DeepSeek validates
   `vocab_size % mesh_cols == 0` (`lm_head1d.py:96-99`). Our mesh is
   `(1, 4)`, so 152064/4 = 38016 ✓. If we later move to `(2, 2)`, only
   2 shards instead of 4 — code still works, but per-chip slab doubles.

---

## Effort estimate (maintainer shipping tomorrow)

| Task | LOC | Time |
|---|---|---|
| `MeshServerState` field additions | 3 | 2 min |
| Bootstrap Change 1 (sharded upload + asserts) | 8 | 10 min |
| Forward Change 2 (3-tuple return + trace plumbing) | 12 | 25 min |
| `_resolve_sharded_argmax` helper | 15 | 15 min |
| Output read Change 3 | 4 | 5 min |
| Sampler gate in `handle_generate_tp` (optional) | 10 | 15 min |
| Validation probe `p_lm_head_sharded_argmax.py` | 80 | 45 min |
| qb2 run + analysis | — | 30 min |
| **Total** | **~50 LOC** | **~2.5 hours** |

The 50-LOC ballpark matches the menu's "lowest effort, ~50 LOC" claim
(`reference_multi_chip_opt_menu.md:11, 19`).

---

## Sequencing relative to other multi-chip optimizations

- **Independent of** mesh paged_sdpa (candidate #13, `feedback_p1_sdpa_decode_breaks_on_mesh.md`).
- **Independent of** distributed RMSNorm (#1, `reference_multi_chip_opt_menu.md:14`)
  — order doesn't matter; they touch different code paths.
- **Stacks with** all_gather_concat (#16, addendum) and llama_rs_matmul (#17)
  for the attention + MLP path; lm_head is the *final* op so it doesn't
  collide.
- **Sampler path:** if D'3 speculative decoding ships
  (`feedback_speculative_decoding.md`), the verify step still uses argmax
  per draft token — the same fast path works for B=2 because per-chip
  argmax is independent.
- **Pre-deepseek refactor:** if we later move to the DeepSeek
  `RunDecodeConfig`-style config-dataclass pattern across the server,
  `lm_head1d.py` is the cleanest place to copy whole. For the
  ship-tomorrow plan above we stay in the existing imperative style.

---

## Open questions to resolve at integration time

1. **Does `ttnn.argmax` work over a sharded `[1, 38016]` on (1,4) mesh in
   our build?** Galaxy uses it after `line_all_gather` makes the tensor
   replicated; we want it on the *unaggregated* per-chip slab. The op
   itself is per-chip-local, so it should — but the
   `feedback_p1_sdpa_decode_breaks_on_mesh.md` precedent (decode-mode ops
   sometimes don't accept mesh tensors) means we MUST verify before
   capture.

2. **Does the matmul output land in L1 or DRAM by default?**
   `feedback_p3_lm_head_replicated_pass.md` doesn't say; DeepSeek decode
   uses `ttnn.L1_MEMORY_CONFIG` (`lm_head1d.py:156`). If our default
   `ttnn.linear` lands in L1 and overflows for the per-chip 38016-wide
   output, force `memory_config=ttnn.DRAM_MEMORY_CONFIG`.

3. **Math fidelity.** DeepSeek uses `COMPUTE_KERNEL_CONFIG_HIFI2`
   (`lm_head1d.py:157`). We don't pass `compute_kernel_config` to the
   current `ttnn.linear(x_tt, state.lm_head_tt)` (`server_tp.py:673`),
   which means we get default fidelity. Since the prod baseline already
   passes correctness with default, leave it untouched at first.

---

## Citations summary

- DeepSeek pattern: `experiments/.refs/tt-metal/models/demos/deepseek_v3/tt/lm_head1d.py:38-262`
- Galaxy pattern (compare): `experiments/.refs/tt-metal/models/demos/llama3_70b_galaxy/tt/lm_head.py:11-193`
- Galaxy on-device argmax in trace: `experiments/.refs/tt-metal/models/demos/llama3_70b_galaxy/tt/llama_model.py:607` and `demo_performance.py:244, 278`
- Our current replicated path: `experiments/serve/server_tp.py:277, 673, 837-839`
- Prior P3 probe (replicated works, sharding projected 3 ms savings):
  `feedback_p3_lm_head_replicated_pass.md:13-25`
- Single-chip `ttnn.argmax` validation: `experiments/55_readback_optimization.py:32-65`
- Menu v2 addendum (DeepSeek beats Galaxy as anchor):
  `reference_multi_chip_opt_menu_v2.md:17`
- Original menu (#6, ~50 LOC, low risk):
  `reference_multi_chip_opt_menu.md:11, 19`
