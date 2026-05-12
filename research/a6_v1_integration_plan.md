# A6 v1 → 91l Prefill Integration Plan

**Goal.** Replace the per-token sequential prefill loop in
`experiments/91l_fp32_residual_generate.py` (lines 248–253) with a chunked
prefill path that drives DeltaNet layers through the Phase A6 v1 chunked-serial
kernel from `experiments/85_deltanet_scan_v1.py`. Full-attention layers
(every 4th layer) continue to run one token at a time inside the chunk.

**Scope.** Prefill only. Decode loop (lines 326–338) is untouched and continues
to call the existing single-token `deltanet_step_ondevice` and
`gated_attn_step_ondevice`. No multi-chip work, no batched-SDPA prefill, no
MAX_POS scaleup (that is C'0.5 owned by another agent).

---

## 1. What A6 v1 actually exposes (read of `85_deltanet_scan_v1.py`)

The script does **not** export a function that consumes the production
`w_tt` dict. It is a standalone correctness/perf harness. The reusable
primitive is the body of `_deltanet_step_on_device` (lines 83–115) lifted into
a serial scan loop (lines 118–136).

### Signatures present

```python
_deltanet_step_on_device(q, k, v, g, beta, H, ttnn)
    # in : q,k [B,H,d_k], v [B,H,d_v], g,beta [B,H], H [B,H,d_k,d_v]
    # out: out [B,H,d_v], H_new [B,H,d_k,d_v]

deltanet_scan_ttnn_v1(Q_dev, K_dev, V_dev, G_dev, BETA_dev, H_init_dev,
                      T, device, ttnn)
    # in : Q_dev … BETA_dev are *lists of T device tensors* (one per timestep)
    #      H_init_dev [B,H,d_k,d_v]
    # out: outs (list of T device tensors), H_final
```

### Important properties

| Property | Value |
|---|---|
| Shape model | per-timestep slices uploaded individually |
| Tensor sequence axis | None — caller materialises a Python list of length T |
| H state dtype | `ttnn.float32`, shape `[B, N_V_HEADS, D_K, D_V]` |
| Q/K/V/g/beta dtype | `ttnn.bfloat16` (script default) |
| Q/K L2-normalisation | inside the kernel |
| Q scaling by `1/sqrt(K_DIM)` | **NOT applied** (91f's `deltanet_step_ondevice` does this — see 91f:176) |
| dt_bias / A_log path | **NOT inside** A6 v1 — g and beta are passed in pre-computed |
| Conv1d state | **NOT handled at all** — A6 v1 starts from clean Q/K/V/g/beta |
| Pre/post norms, projections | **NOT included** |
| Output projection, residual add | **NOT included** |

**This is the central finding:** A6 v1 covers only the recurrence inner loop
(the H-evolution and out=qH math). Everything else in
`deltanet_step_ondevice` — RMSNorm, in_proj_qkv/z/a/b linears, causal conv,
softplus/A_log/g/beta computation, per-head gated RMSNorm, out_proj, residual
add — must still happen **per-chunk-position**, just like it does per-token
today.

---

## 2. Read of `deltanet_step_ondevice` (91f:122-214)

The single-token path does, in order:

1. `rms_norm(x)` (input_layernorm)
2. Four `linear`s → `mixed_qkv` [CONV_DIM], `z`, `a`, `b` [VAL_DIM or N_V]
3. 1-D causal conv via `concat([conv_state, mixed_col], dim=-1)` then `mul(weight) + sum + silu` → `conv_out` [CONV_DIM]
4. Slice conv_out into q_flat, k_flat, v_flat; GQA-interleave to N_V_HEADS
5. L2-normalise q,k; scale q by `1/sqrt(K_DIM)`
6. `g = -exp(A_log) * softplus(a + dt_bias)`; `beta = sigmoid(b)` — these are the (g, beta) A6 v1 expects
7. Recurrence: H decay, kv read, delta, H update, q read → `out` [VAL_DIM]
8. Per-head RMSNormGated with `linear_attn_norm` and `silu(z)`
9. `out_proj`, residual add `x_out = x + out_proj`
10. Return `x_out`, `H_new` (reshaped to `[N_V_HEADS, K_DIM, V_DIM]`), `conv_state_new`

A6 v1 covers only step 7. Steps 1–6 and 8–10 must be redone per chunk
position (they have no time-step dependency past conv_state, which is local).

---

## 3. The API gap

| | A6 v1 (`_deltanet_step_on_device`) | 91l current call |
|---|---|---|
| Inputs | q,k,v,g,beta (post-projection) | x_tt + w_tt + states |
| Length | 1 step at a time, but driven from a T-list | 1 token |
| Conv state | not modelled | `[CONV_DIM, K_conv-1]`, evolved internally |
| H shape | `[B, N_V_HEADS, D_K, D_V]` | `[N_V_HEADS, K_DIM, V_DIM]` |
| Residual add | not included | `x + out_proj` |
| Out gating + out_proj | not included | included |

Conclusion: we cannot drop A6 v1 in as a black-box layer step. We need a new
function — call it **`deltanet_chunk_ondevice`** — that wraps:
**pre-recurrence per-position projections (steps 1–6)** + **A6 v1's serial
H-scan over the C positions** + **per-position post-recurrence (steps 8–10)**.

The benefit over per-token comes from:
- Eliminating Python-side per-token JIT overhead by batching the C linears as one matmul over `[C, HIDDEN]` (the 4 input projections become one `linear(x[C,H], w)` returning `[C, OUT]`).
- Eliminating C round-trips through the rest of 91l's `forward_one_token` machinery.
- Keeping the recurrence on-device throughout the C steps (no host sync).

The 800 tok/s figure in `phase_a6_a7_results.md` was measured for the inner
recurrence only at C=1024. The full DeltaNet-chunk path will be somewhat
slower than that, but still >>5-10x the current per-token rate.

---

## 4. Concrete edit plan for `91l`

### 4a. New module `experiments/utils/deltanet_chunk.py`

```python
def deltanet_chunk_ondevice(x_chunk_tt, w_tt, ssm_state_tt, conv_state_tt, cfg,
                            ttnn, device):
    """
    Chunked DeltaNet step.

    Inputs:
      x_chunk_tt : [C, HIDDEN] (fp32 residual stream slice for this chunk)
      w_tt       : same dict as 91l layer_weights[i]
      ssm_state_tt: [N_V_HEADS, K_DIM, V_DIM] fp32
      conv_state_tt: [CONV_DIM, K_conv-1]

    Returns:
      x_out_chunk_tt : [C, HIDDEN]  (residual stream after this layer)
      ssm_state_new  : same shape as input
      conv_state_new : same shape as input
    """
```

**Internals** (C = chunk length):

1. `h = rms_norm(x_chunk_tt, w['input_layernorm'])` → `[C, HIDDEN]`
2. Single batched linear per projection: `mixed_qkv = linear(h, w['in_proj_qkv'])` → `[C, CONV_DIM]`; same for z, a, b.
3. Conv: append `mixed_qkv` to `conv_state_tt` along the time axis to form a `[CONV_DIM, K_conv-1 + C]` working buffer. For each `t in range(C)`: slice `[:, t : t+K_conv]`, multiply by `w['conv1d_weight']`, sum-reduce, silu → `conv_out_t` [CONV_DIM]. After the loop, the new conv_state is the **last K_conv-1 columns** of the working buffer.
4. For each `t`: do the slice/interleave/normalise/scale to get q_t, k_t, v_t, then compute g_t, beta_t from `a[t]`, `b[t]` (and the per-layer A_log, dt_bias scalars). Call A6 v1's `_deltanet_step_on_device(q_t, k_t, v_t, g_t, beta_t, H, ttnn)` → updates H and emits `out_t`.
5. Stack `out_t` into `[C, VAL_DIM]` (the A6 v1 paper-stack is just keeping the list).
6. Per-position post: reshape to `[C * N_V_HEADS, V_DIM]` for the gated RMSNorm against `linear_attn_norm` weight (which broadcasts over C), gate with `silu(z[C, VAL_DIM])` reshaped likewise, reshape back to `[C, VAL_DIM]`.
7. `out_proj` (one `[C, VAL_DIM] @ [VAL_DIM, HIDDEN]`), then `x_out_chunk = x_chunk_tt + out_proj`.
8. Return `x_out_chunk_tt`, `H` (final state), conv_state new.

**Note** the only real "scan" is the H update loop. Steps 1, 2, 6, 7 are all
batched over C. Step 3 is C scalar ops on small tensors (CONV_DIM=2KEY+VAL ≈
6k). Step 4 is C calls into the A6 v1 step. This is exactly the same arithmetic
as the existing per-token path, just amortising Python+JIT overhead.

### 4b. Modifications to `91l_fp32_residual_generate.py`

Add CLI flag and constant:

```python
p.add_argument('--prefill-mode', choices=['serial', 'chunked'], default='chunked')
p.add_argument('--chunk-size', type=int, default=64)
```

Add `forward_chunk(prompt_token_ids, start_pos)` next to `forward_one_token`:

```python
def forward_chunk(token_ids, start_pos):
    C = len(token_ids)
    # 1. Embed C tokens at once → [C, HIDDEN] bf16
    x_np = embed_np[token_ids]                  # [C, HIDDEN]
    x_tt = upload(x_np, device, dtype=ttnn.bfloat16)

    # 2. Per-position RoPE tables for partial-RoPE full-attn layers.
    cos_list, sin_list, pos_tt_list = [], [], []
    for k in range(C):
        cos_k, sin_k = rope_tables_for_pos(start_pos + k)
        cos_list.append(cos_k); sin_list.append(sin_k)
        pos_tt_list.append(ttnn.from_torch(
            torch.tensor([start_pos + k], dtype=torch.int32), device=device))

    dn_idx = 0; attn_idx = 0
    for i in range(NUM_LAYERS):
        layer_type, w_tt = layer_weights[i]
        if layer_type == 'linear_attention':
            x_tt, H_new, c_new = deltanet_chunk_ondevice(
                x_tt, w_tt, ssm_states[dn_idx], conv_states[dn_idx],
                cfg, ttnn, device)
            ssm_states[dn_idx] = H_new
            conv_states[dn_idx] = c_new
            dn_idx += 1
        else:
            # Full-attention: serial over C positions, kv cache writes are
            # the same as decode (per-position scatter).
            kv_k, kv_v = kv_caches[attn_idx]
            for k in range(C):
                row_tt = ttnn.slice(x_tt, [k, 0], [k+1, HIDDEN])
                row_tt, kv_k, kv_v = gated_attn_step_ondevice(
                    row_tt, w_tt, kv_k, kv_v, None,
                    pos_tt_list[k], start_pos + k,
                    cos_list[k], sin_list[k], cfg, device)
                # write row_tt back into x_tt[k]
                x_tt = scatter_row(x_tt, row_tt, k)   # helper, see note
            kv_caches[attn_idx] = [kv_k, kv_v]
            attn_idx += 1
        x_tt = mlp_step_ondevice(x_tt, w_tt)  # mlp is per-row, batches over C
    # No lm_head call here — caller decides what to do with x_tt.
    return x_tt
```

`scatter_row` is implemented as `concat([x_tt[:k], row_tt, x_tt[k+1:]], dim=0)`
since ttnn lacks in-place tensor assignment. (Acceptable cost: C concats per
full-attn layer; 16 full-attn layers × 64 = 1024 concats over a 64-chunk
prefill — small vs the matmul cost.)

Replace lines 249–253 of `91l` with:

```python
if args.prefill_mode == 'serial':
    for pos, tid in enumerate(prompt_ids):
        _ = forward_one_token(tid, pos)
else:
    C = args.chunk_size
    for start in range(0, len(prompt_ids), C):
        chunk = prompt_ids[start:start+C]
        _ = forward_chunk(chunk, start)
```

For the **last prefill chunk**, we additionally need the logits at the last
position so decode can take over. Two options:
- (preferred) Apply `final_norm + lm_head` on the **last row only** of the
  chunk's `x_tt` after the loop. The decode loop's first iteration then becomes
  redundant for the first generated token, which is fine.
- Or fall through to one more `forward_one_token(prompt_ids[-1], len(prompt_ids)-1)`
  call — wastes one token but simplest. **Not** what we should do because it
  double-writes that token's KV slot and double-evolves the SSM state. **Use
  option 1.**

---

## 5. Conv-state continuity across chunks

The 1-D causal conv has `K_conv = 4` (Qwen3.6 `linear_conv_kernel_dim`).
Inside one chunk of length C, the working conv buffer is
`[CONV_DIM, K_conv-1 + C] = [CONV_DIM, 3 + C]`. The new conv state is
`buf[:, -(K_conv-1):] = buf[:, -3:]`. This is exactly what the single-token
path does at C=1: `slice(conv_input, [0, 1], [CONV_DIM, KERNEL])` (91f:144).
The chunked version generalises that to: after concatenating C new columns,
keep the trailing 3.

A6 v1 itself does **not** handle conv state — this entire mechanism lives in
`deltanet_chunk_ondevice`, not in the lifted A6 v1 step.

---

## 6. Test plan

| Test | File | Pass criterion |
|---|---|---|
| **Unit, sequential parity** | `experiments/86_deltanet_chunk_unit.py` (new) | Random N∈{8,64,65,128,256,1024}, fixed seed. Compare `deltanet_chunk_ondevice(x[:N])` to N calls to `deltanet_step_ondevice`. Per-position cosine ≥ 0.9996. Final ssm_state cosine ≥ 0.9996. Final conv_state cosine ≥ 0.9996. |
| **Boundary**, C=65 with chunk=64 | same file | Two chunks (64 + 1) match one chunk of 65. Validates state hand-off. |
| **Real prompt logits parity** | `experiments/87_chunk_prefill_logits.py` (new) | Tokenize "The capital of France is" (5 tokens). Run `91l` serial prefill, capture last-position logits. Run chunked with chunk_size=8 (still C>=N so single chunk). Top-5 must be identical and per-logit max abs diff < 1e-3. |
| **Paris demo** | `91l --prefill-mode chunked --tokens 60` | "Paris" still appears as first generated token; first 30 tokens are coherent. |
| **Per-layer drift** | extend `91r` with `--prefill-mode chunked` | For each of the default layers, compare chunked-prefill output of layer i to HF hidden_states[i+1]. All ≥ 0.9997. |
| **Throughput** | `91l --prefill-mode chunked` instrumented | At N=512, prefill total time ≤ N / 600 sec (target 600+ tok/s including full-attn serial-within-chunk). |

---

## 7. Backward compat

- `deltanet_step_ondevice` (91f:122) is **untouched**. Decode loop continues to call it.
- New `deltanet_chunk_ondevice` lives in `experiments/utils/deltanet_chunk.py`.
- `91l` is the only consumer; it gains a `--prefill-mode` flag whose default
  is `chunked`. Setting `--prefill-mode serial` reproduces today's behaviour
  bit-for-bit.

---

## 8. Risks

1. **Numerical regime mismatch.** Per-token path uses bf16 residual, fp32 SSM
   state; the batched linears over C rows may take a slightly different
   bf16/HiFi4 reduction order. Mitigate: enforce hifi4 + fp32_dest_acc_en on
   every matmul in the chunk path (same kernel config as 91l line 51).
2. **Conv-state continuity bug.** Easy to fencepost: K_conv-1=3, not K_conv=4.
   Test C=65 with chunk=64 explicitly to catch this.
3. **Q-scaling missing.** A6 v1 omits `q *= 1/sqrt(K_DIM)`. Must be added in
   `deltanet_chunk_ondevice` step 4, mirroring 91f:176. If we forget, cosine
   silently drops on layers ≥ 2 (same failure mode documented in B'9.5).
4. **GQA interleave semantics.** 91f:155-158 uses `unsqueeze + repeat-singleton
   + flatten` to get repeat_interleave semantics. Chunked path must use the
   same — `ttnn.repeat` is tile-semantics, not interleave. Reuse the helper.
5. **scatter_row cost / memory.** 16 concats × 64 positions × 64 layers of
   prefill = ~65k concats; each is tiny but might dominate. If profiling
   shows this is the bottleneck, replace full-attn-inside-chunk with a single
   batched SDPA-prefill (out of scope — call it C'5b).
6. **Last-chunk logits handoff.** Need to slice the last row out of the chunk's
   final x_tt before lm_head; an off-by-one here gives the wrong first decode
   token but does not affect the chunked code itself.

---

## 9. Effort estimate

| Task | Hours |
|---|---|
| Write `deltanet_chunk_ondevice` | 3 |
| Unit test 86 + boundary | 2 |
| 91l integration + scatter_row helper + last-chunk logits | 2 |
| Real-prompt parity test 87 | 1 |
| 91r `--prefill-mode chunked` extension | 1 |
| Debug (Q-scaling / conv fencepost / GQA / dtype regressions) | 4 |
| **Total** | **~13h** (1.5 working days) |

Throughput target: 5-token prompt 80 ms; 512-token prompt ≤ 0.9 s; 32k prompt
~55 s (vs 80 min today).
