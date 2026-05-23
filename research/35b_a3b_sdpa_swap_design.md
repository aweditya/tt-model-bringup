# SDPA swap design for `server_35b_ttnn.py:attn_forward_ttnn`

Design memo for replacing the manual `matmul → softmax → matmul` attention
path with `ttnn.transformer.paged_scaled_dot_product_attention_decode` +
B3 compute_kernel_config. Same pattern as 27B server_tp.py.

**Pre-write rule (CLAUDE.md non-negotiables):** every shape claim here is
checked against a specific source line. No hand-waving.

## What's in scope

| Touch | File / lines | Why |
|-------|--------------|-----|
| `State.__init__` | server_35b_ttnn.py:841-859 | Add fields for page_table_tt, paged_write_mem_cfg, paged_sdpa_progcfg, sdpa_compute_kernel_config, attn_mode flag |
| `reset_caches_ttnn` | :861-883 | Allocate paged KV cache per attn layer (4D `[NUM_BLOCKS, 1, BLOCK_SIZE, HEAD_DIM]`) instead of None |
| `bootstrap` | :1066+ | Set up page_table, mem_cfg, progcfg, kernel_config |
| `attn_forward_ttnn` | :568-690 | Add a `sdpa` branch under `state.attn_mode`; keep `manual` branch as `attn_forward_ttnn_manual` for fallback |

## Out of scope

- DN block — not the drift source per `feedback_35b_a3b_attn_layer_drift.md`.
- MoE block — not the drift source.
- B17 trace work — must verify trace still captures, but no changes to trace surface beyond what the SDPA swap requires.
- RoPE — keep the broadcast workaround for K (the `[1, HEAD_DIM]` ttnn bug from `feedback_qwen36_attn_rope_single_row_ttnn_bug.md` still applies). We dedupe back to `[1, HEAD_DIM]` AFTER RoPE before writing to cache. V doesn't need RoPE so V skips the broadcast.

## Shapes (35B per chip, NCHIPS=4)

Constants from server_35b_ttnn.py:
- `NUM_Q_HEADS = 16`, `NQ_PER_CHIP = 4`
- `NUM_KV_HEADS = 2`, `chips_per_kv = 2` → 1 KV head per chip (chips 0,1 share KV head 0; chips 2,3 share KV head 1)
- `HEAD_DIM_ATTN = 256`
- `HIDDEN = 2048`
- `EPS` for rms_norm
- MAX_KV = 4096 (from cos/sin table size, server log line `"cos/sin tables (4096 positions) ready"`)

Per chip during decode (after q_proj, k_proj, v_proj):
- `q_full`: `[1, NQ_PER_CHIP * HEAD_DIM * 2] = [1, 2048]`  (Q + gate concatenated)
- `k`: `[1, HEAD_DIM] = [1, 256]`
- `v`: `[1, HEAD_DIM] = [1, 256]`

After per-head Q/gate split + rms_norm:
- `q_n`: `[NQ_PER_CHIP, HEAD_DIM] = [4, 256]`
- `k_n`: `[1, HEAD_DIM] = [1, 256]` (intermediate, before broadcast)

After RoPE (with broadcast workaround for K):
- `q_n`: `[NQ_PER_CHIP, HEAD_DIM] = [4, 256]` — direct rotation works
- `k_n` (broadcast): `[NQ_PER_CHIP, HEAD_DIM] = [4, 256]` — all 4 rows identical
- **NEW**: dedupe back to `[1, HEAD_DIM]` by taking row 0 (math-equivalent because rotation is identical across the 4 copies). Use `ttnn.slice` on dim 0 with [0:1, :].

## Paged cache shape

Per chip:
- `BLOCK_SIZE = 32` (must be multiple of TILE_HEIGHT)
- `NUM_BLOCKS = MAX_KV / BLOCK_SIZE = 4096 / 32 = 128`
- K cache: `[NUM_BLOCKS=128, 1, BLOCK_SIZE=32, HEAD_DIM=256]` bf16 in DRAM
- V cache: same shape
- `page_table_tt`: `[1, NUM_BLOCKS] = [1, 128]` int32 identity (logical block i → physical block i for B=1)

## SDPA call signature (per chip)

```python
# After RoPE + K dedupe:
k_n_single = ttnn.slice(k_n, [0, 0], [1, HEAD_DIM_ATTN])  # [1, 256]
ttnn.deallocate(k_n)  # free broadcast copy
v_h = ttnn.reshape(v, [1, HEAD_DIM_ATTN])                 # [1, 256]
ttnn.deallocate(v)

# Reshape to 4D HEIGHT_SHARDED-input contract: [1, 1, NKV_PER_CHIP=1, HEAD_DIM]
# then pad dim -2 to TILE_HEIGHT=32, then HEIGHT_SHARDED L1
def _shard_for_paged_write(t_2d, mem_cfg):
    t4d = ttnn.reshape(t_2d, [1, 1, 1, HEAD_DIM_ATTN])
    t_pad = ttnn.pad(t4d, [[0,0],[0,0],[0, TILE_HEIGHT-1],[0,0]], value=0.0)
    return ttnn.to_memory_config(t_pad, mem_cfg)

k_sharded = _shard_for_paged_write(k_n_single, state.paged_write_mem_cfg)
v_sharded = _shard_for_paged_write(v_h, state.paged_write_mem_cfg)
ttnn.experimental.paged_update_cache(kc, k_sharded,
    update_idxs_tensor=state.cur_pos_buf, page_table=state.page_table_tt)
ttnn.experimental.paged_update_cache(vc, v_sharded,
    update_idxs_tensor=state.cur_pos_buf, page_table=state.page_table_tt)

# SDPA decode
q_for_sdpa = ttnn.reshape(q_n, [1, 1, NQ_PER_CHIP, HEAD_DIM_ATTN])  # [1, 1, 4, 256]
attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
    q_for_sdpa, kc, vc,
    cur_pos_tensor=state.cur_pos_buf,
    page_table_tensor=state.page_table_tt,
    program_config=state.paged_sdpa_progcfg,
    compute_kernel_config=state.sdpa_compute_kernel_config,
)  # [1, 1, NQ_PER_CHIP, HEAD_DIM] per chip
attn_per_head = ttnn.reshape(attn_out, [NQ_PER_CHIP, HEAD_DIM_ATTN])
```

## Configs (B3 recipe, same as 27B)

```python
state.paged_sdpa_progcfg = ttnn.SDPAProgramConfig(
    compute_with_storage_grid_size=ttnn.CoreCoord(4, 4),
    q_chunk_size=0,
    k_chunk_size=0,
    exp_approx_mode=False,
)
state.sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=False,
)
```

## State.cur_pos_buf

server_35b_ttnn currently passes `pos` (Python int) to update_input_buffers
which writes only `tok_buf` and `rot_idxs_buf`. paged SDPA + paged_update_cache
require a **device tensor** for cur_pos_tensor. We add:
- `state.cur_pos_buf` = `[1] int32` device tensor (replicated across mesh)
- `update_input_buffers` writes the cur_pos to it via copy_host_to_device

This is a small extension to the existing input-buffer pattern.

## Flag and fallback

```python
class State:
    def __init__(self):
        ...
        self.attn_mode = "sdpa"   # default after this swap
        # manual path kept as fallback for A/B and rollback safety
```

`attn_forward_ttnn(...)` dispatches:
```python
def attn_forward_ttnn(h_tt, w, mesh, cos_tt, sin_tt, kv_cache=None, *,
                      state=None):
    if state is not None and state.attn_mode == "sdpa":
        return attn_forward_ttnn_sdpa(h_tt, w, mesh, cos_tt, sin_tt, state=state)
    return attn_forward_ttnn_manual(h_tt, w, mesh, cos_tt, sin_tt, kv_cache=kv_cache)
```

`layer_forward_ttnn` needs to thread `state` through to `attn_forward_ttnn`.
That's a small signature change touching ~3 call sites.

## Trace compatibility

paged_update_cache + paged_scaled_dot_product_attention_decode are both
trace-friendly (verified at qb2 prod commit `4741253`). The trace boundary
is the same as today — update_input_buffers writes (now including cur_pos)
OUTSIDE trace, step_forward_inner runs inside trace.

## Pre-flight checklist before refactor

- [ ] qb1 has 35B-A3B weights cached (yes, just verified)
- [ ] qb1 mesh accessible, no resident server (yes, currently free)
- [ ] HF oracle exists at `.cache/hf_oracle_35b_100tok/` for ladder comparison
- [ ] Baseline (manual) numbers captured in `feedback_35b_a3b_attn_layer_drift.md` (yes)
- [ ] cosine_ladder_35b.py exists and works (yes)
- [ ] State.attn_mode flag will let us A/B without touching B17 trace prod

## Risks

1. **NUM_BLOCKS=128 cache footprint**: per-chip K = 128*1*32*256*2 bytes = 2 MB; V same. 10 attn layers × (2+2) MB = 40 MB per chip. Well within DRAM budget.
2. **Shard grid mismatch**: 27B uses CoreCoord(4,4); 35B may have a different worker grid. Verify with mesh inspection at bootstrap.
3. **Dedupe of K after RoPE**: ttnn.slice on [4, 256] tile-layout taking [0:1, :] should be tile-aligned and safe per `feedback_ttnn_slice_row_aligned.md`. Verify.
4. **HEIGHT_SHARDED L1 input pad**: pad from `[1, 1, 1, 256]` to `[1, 1, 32, 256]` (TILE_HEIGHT). Standard pattern.
5. **MAX_KV vs MAX_POS naming**: 35B uses "MAX_KV"; 27B uses "MAX_POS". Same semantic.
6. **State threading change**: layer_forward_ttnn signature grows by 1 arg. Must update call sites (~3) without breaking trace.
7. **The smoke prompt is 85 tokens; MAX_KV=4096 is plenty** — but verify cache wrap-around behavior just to be safe.

## Open questions / risks I want to flag to user BEFORE coding

1. **Worker grid size** on 35B (1,4) mesh — qb1 has 120 Tensix cores per chip after firmware downgrade (`feedback_p150_firmware_core_check.md`). 27B's CoreCoord(4,4)=16 cores. Probably fine for 35B too but verify.
2. **The K-dedupe-after-RoPE step is new**. We're assuming all 4 copies of K stay identical post-RoPE because cos/sin are scalar per step. Math says yes. But empirically — should I add a debug assertion that compares row 0 to row 1 of the broadcast K?
3. **B17 trace bootstrap**: the trace was captured against the manual attn path. Adding a flag means trace must be re-captured against the SDPA path (since the kernel sequence differs). The `attn_mode` flag must be set BEFORE bootstrap/trace capture.
4. **`update_input_buffers` signature change**: takes (state, token_id, cur_pos) currently. With cur_pos_buf added, the write goes into state.cur_pos_buf via copy_host_to_device. Existing trace-outside writes already happen here; we just add one more.

## Implementation order

1. Add State fields + bootstrap setup (NO behavior change yet; manual still default — actually make `attn_mode = "manual"` initially so prod path is untouched until we test SDPA)
2. Implement `attn_forward_ttnn_sdpa` as a new function
3. Add dispatcher in `attn_forward_ttnn`
4. Thread state through layer_forward_ttnn → step_forward_inner
5. Update update_input_buffers to write cur_pos_buf
6. Rerun smoke (--n-positions 10) under sdpa mode; expect significant improvement in AT-layer cosines
7. If smoke passes → full 85-pos ladder
8. Compare against manual baseline
9. If success → flip default to "sdpa" in a follow-up commit
