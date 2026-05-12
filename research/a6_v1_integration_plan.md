# A6 v1 → 91l Prefill Integration Plan

**Goal.** Replace the per-token prefill loop in
`experiments/91l_fp32_residual_generate.py` (lines 248–253) with a chunked path
that drives DeltaNet layers through the A6 v1 kernel from
`experiments/85_deltanet_scan_v1.py`. Full-attention layers (every 4th) run
one token at a time inside the chunk.

**Scope.** Prefill only. Decode loop (91l:326–338) untouched. No multi-chip,
no batched-SDPA prefill, no MAX_POS scaleup (C'0.5 elsewhere).

---

## 1. What A6 v1 actually exposes (`85_deltanet_scan_v1.py`)

Standalone harness — does not export anything consuming 91l's `w_tt`. The
reusable primitive is `_deltanet_step_on_device` (83–115) inside the scan
`deltanet_scan_ttnn_v1` (118–136).

Signatures:

```python
_deltanet_step_on_device(q, k, v, g, beta, H, ttnn)
    # in : q,k [B,H,d_k], v [B,H,d_v], g,beta [B,H], H [B,H,d_k,d_v]
    # out: out [B,H,d_v], H_new [B,H,d_k,d_v]

deltanet_scan_ttnn_v1(Q_dev, K_dev, V_dev, G_dev, BETA_dev, H_init_dev,
                      T, device, ttnn)
    # Q_dev…BETA_dev are LISTS of T device tensors. H init: fp32. Returns list of T outs + H_final.
```

Properties:

| Property | Value |
|---|---|
| Sequence representation | Python list of per-step uploads (no T-axis tensor) |
| H dtype/shape | `float32`, `[B, N_V_HEADS, D_K, D_V]` |
| Q/K/V/g/beta dtype | bf16 (harness) |
| Q,K L2 normalise | yes, inside kernel |
| Q `* 1/sqrt(K_DIM)` scaling | **NOT applied** (91f:176 has it) |
| `g = -exp(A_log)*softplus(a+dt_bias)`, `beta = sigmoid(b)` | **NOT inside** — caller pre-computes |
| Conv1d state | **NOT handled** |
| Pre/post norms, projections, out_proj, residual add | **NOT included** |

**Central finding:** A6 v1 covers only the H-recurrence inner loop.
Everything else in `deltanet_step_ondevice` (rms_norm, in_proj_*, causal conv,
g/beta, gated RMSNorm, out_proj, residual add) must still happen
per-chunk-position.

---

## 2. `deltanet_step_ondevice` per-token path (91f:122-214)

Order: (1) `rms_norm(x)`; (2) four `linear`s → `mixed_qkv`, `z`, `a`, `b`;
(3) causal conv via `concat([conv_state, mixed_col], -1)` then `mul(w)+sum+silu`
→ `conv_out [CONV_DIM]`; (4) slice into q,k,v_flat + GQA-interleave to N_V_HEADS;
(5) L2-norm q,k + scale `q *= 1/sqrt(K_DIM)`; (6) `g = -exp(A_log)*softplus(a+dt_bias)`,
`beta = sigmoid(b)`; (7) **recurrence — A6 v1's territory**; (8) per-head
RMSNormGated with `linear_attn_norm` and `silu(z)`; (9) `out_proj`,
`x_out = x + out_proj`; (10) return `x_out`, `H_new` (`[N_V_HEADS,K_DIM,V_DIM]`),
`conv_state_new`.

A6 v1 covers only step 7. Steps 1–2, 4–6, 8–9 are per-position pointwise — they
batch over C rows. Step 3 carries `conv_state`; step 7 carries `H`.

## 3. Current 91l prefill (91l:248-253)

```python
for pos, tid in enumerate(prompt_ids):
    _ = forward_one_token(tid, pos)
```

`forward_one_token` (91l:208-246) embeds one token, builds per-pos RoPE,
loops 64 layers calling `deltanet_step_ondevice` (state:
ssm_states/conv_states) or `gated_attn_step_ondevice` (state: kv_caches),
then `mlp_step_ondevice`, then `final_norm + lm_head`. State lists are
mutated in place from `main()` scope.

---

## 4. The API gap

| | A6 v1 `_deltanet_step_on_device` | 91l call site |
|---|---|---|
| Inputs | q,k,v,g,beta (post-projection) | x_tt, w_tt, states |
| Conv state | not modelled | `[CONV_DIM, K_conv-1]`, internal |
| Length | 1 step (driven from T-list) | 1 token |
| H shape | `[B, N_V_HEADS, D_K, D_V]` | `[N_V_HEADS, K_DIM, V_DIM]` |
| Residual add, out_gating, out_proj | none | included |

A6 v1 cannot drop in as a black-box layer step. Wrap it in
**`deltanet_chunk_ondevice`** that does steps 1–2 batched over C, step 3
extending conv_state by C columns, steps 4–6 vectorised, step 7 as a C-length
call to A6 v1's inner step, and steps 8–9 batched.

Speedup vs per-token: (a) 4 input projections become 1 matmul each over
`[C,HIDDEN]` instead of C matmuls; (b) Python/JIT overhead amortised C-fold;
(c) recurrence stays on-device, no host sync.

---

## 5. Concrete edit plan for 91l

**New module** `experiments/utils/deltanet_chunk.py`:

```python
def deltanet_chunk_ondevice(x_chunk_tt, w_tt, ssm_state_tt, conv_state_tt, cfg,
                            ttnn, device):
    """
    x_chunk_tt   : [C, HIDDEN]  fp32 residual slice for this chunk
    w_tt         : same dict shape as 91l layer_weights[i]
    ssm_state_tt : [N_V_HEADS, K_DIM, V_DIM] fp32
    conv_state_tt: [CONV_DIM, K_conv-1]
    Returns x_out_chunk_tt [C, HIDDEN], ssm_state_new, conv_state_new
    """
```

Internals: (1) `h = rms_norm(x_chunk_tt, w['input_layernorm'])` → `[C,HIDDEN]`.
(2) Four batched linears with hifi4 (91l:51 kernel config). (3) Conv buffer
`concat([conv_state, mixed_qkv.T], -1)` shape `[CONV_DIM, 3+C]`. For
`t in range(C)`: slice `[:, t:t+K_conv]`, `mul(w['conv1d_weight'])`, sum dim=-1,
silu → `conv_out_t`. After loop, `conv_state_new = buf[:, -3:]`. (4–6) For
each `t`: slice/GQA-interleave (reuse 91f's `gqa_interleave` —
unsqueeze+singleton-repeat+flatten, NOT `ttnn.repeat`; 91f:155-158),
L2-normalise q,k, `q *= 1/sqrt(K_DIM)`, compute `g_t, beta_t` from
`a[t], b[t], A_log, dt_bias`. (7) Call
`_deltanet_step_on_device(q_t,k_t,v_t,g_t,beta_t, H, ttnn)` — emits `out_t`,
updates H. (8) Stack outs into `[C, VAL_DIM]`. Reshape `[C*N_V_HEADS, V_DIM]`,
RMSNormGated with `linear_attn_norm`, gate by `silu(z)`, reshape back. (9)
`out_proj` (one matmul); `x_out = x_chunk_tt + out_proj`. Return.

**91l changes** — add CLI flags:

```python
p.add_argument('--prefill-mode', choices=['serial','chunked'], default='chunked')
p.add_argument('--chunk-size', type=int, default=64)
```

Add `forward_chunk(token_ids, start_pos)` next to `forward_one_token`:

```python
def forward_chunk(token_ids, start_pos):
    C = len(token_ids)
    x_tt = upload(embed_np[token_ids], device, dtype=ttnn.bfloat16)  # [C, HIDDEN]
    cos_list, sin_list, pos_tt_list = [], [], []
    for k in range(C):
        c_, s_ = rope_tables_for_pos(start_pos + k)
        cos_list.append(c_); sin_list.append(s_)
        pos_tt_list.append(ttnn.from_torch(
            torch.tensor([start_pos + k], dtype=torch.int32), device=device))
    dn_idx = attn_idx = 0
    for i in range(NUM_LAYERS):
        layer_type, w_tt = layer_weights[i]
        if layer_type == 'linear_attention':
            x_tt, H_new, c_new = deltanet_chunk_ondevice(
                x_tt, w_tt, ssm_states[dn_idx], conv_states[dn_idx],
                cfg, ttnn, device)
            ssm_states[dn_idx] = H_new; conv_states[dn_idx] = c_new; dn_idx += 1
        else:
            kv_k, kv_v = kv_caches[attn_idx]
            for k in range(C):                            # serial within chunk
                row_tt = ttnn.slice(x_tt, [k,0], [k+1, HIDDEN])
                row_tt, kv_k, kv_v = gated_attn_step_ondevice(
                    row_tt, w_tt, kv_k, kv_v, None,
                    pos_tt_list[k], start_pos + k,
                    cos_list[k], sin_list[k], cfg, device)
                x_tt = scatter_row(x_tt, row_tt, k)       # concat-based helper
            kv_caches[attn_idx] = [kv_k, kv_v]; attn_idx += 1
        x_tt = mlp_step_ondevice(x_tt, w_tt)              # batches over C rows
    return x_tt
```

`scatter_row(x, row, k) = concat([x[:k], row, x[k+1:]], dim=0)` (ttnn has no
in-place row write).

Replace 91l:249-253 with:

```python
if args.prefill_mode == 'serial':
    for pos, tid in enumerate(prompt_ids):
        _ = forward_one_token(tid, pos)
else:
    C = args.chunk_size
    last_x = None
    for start in range(0, len(prompt_ids), C):
        last_x = forward_chunk(prompt_ids[start:start+C], start)
    # Pull logits for the LAST prefill position (so decode picks up correctly).
    last_row = ttnn.slice(last_x, [last_x.shape[0]-1, 0], [last_x.shape[0], HIDDEN])
    last_row = ttnn.rms_norm(last_row, weight=final_norm_tt, epsilon=EPS)
    logits_tt = ttnn.linear(last_row, lm_head_tt, compute_kernel_config=hifi4)
    # discard or use as the first decode logits (cleaner: pass into decode loop)
```

## 6. Mixed DeltaNet + full-attention inside a chunk

Pattern: every 4th layer is full_attention (DN,DN,DN,FullAttn,…). Within one
chunk: DN layers run as one chunked call; FullAttn layers iterate C tokens
serially through `gated_attn_step_ondevice`. Cheap because gated_attn is
~2 ms post-C'1, only 16 of 64 layers, and KV writes are per-position anyway.
A future `gated_attn_chunk_ondevice` doing batched SDPA-prefill is **C'5b** —
out of scope.

## 7. Conv-state continuity across chunks

K_conv=4. Working buffer inside one chunk is `[CONV_DIM, K_conv-1 + C] =
[CONV_DIM, 3+C]`; new conv_state = trailing 3 columns. Per-token path does the
same at C=1 (91f:144). A6 v1 does **not** handle conv state — this lives
entirely in `deltanet_chunk_ondevice`.

## 8. Test plan

| Test | Where | Pass criterion |
|---|---|---|
| Sequential parity | new `experiments/86_deltanet_chunk_unit.py` | Random N∈{8,64,65,128,256,1024}; `deltanet_chunk_ondevice` vs N×`deltanet_step_ondevice`; per-pos cosine ≥ 0.9996; final ssm_state and conv_state cosine ≥ 0.9996 |
| Chunk-boundary | same file | N=65 with chunk=64 (two-chunk) equals one-chunk-of-65 |
| Real prompt logits | new `experiments/87_chunk_prefill_logits.py` | "The capital of France is" (5 tok). Last-pos top-5 identical to serial; per-logit max abs diff < 1e-3 |
| Paris demo | `91l --prefill-mode chunked --tokens 60` | First generated token is "Paris"; first 30 tokens coherent |
| Per-layer drift | extend `91r` with `--prefill-mode chunked` | All sampled layers ≥ 0.9997 vs HF hidden_states |
| Throughput | `91l` instrumented | N=512: prefill ≤ 0.9 s (target ≥ 600 tok/s end-to-end including FullAttn-in-chunk) |

## 9. Backward compat

- `deltanet_step_ondevice` (91f:122) is **untouched**; decode keeps using it.
- New code in `experiments/utils/deltanet_chunk.py`.
- 91l gains `--prefill-mode` (default `chunked`). `--prefill-mode serial`
  reproduces today's behaviour bit-for-bit.

## 10. Risks

1. **Q-scaling forgotten.** A6 v1 omits `q *= 1/sqrt(K_DIM)`; the chunked path
   must add it (mirror 91f:176). Missing → cosine drops on layer ≥ 2; same
   failure mode as B'9.5. Catch in unit test (sequential-parity, N=1).
2. **GQA interleave semantics.** `ttnn.repeat` is TILE not interleave; must
   reuse the unsqueeze+singleton-repeat+flatten idiom (91f:155-158).
3. **Conv fencepost.** K_conv-1=3, not K_conv=4. Boundary test C=65 with
   chunk=64 explicitly validates.
4. **Dtype regime drift.** Per-token uses bf16 residual + fp32 SSM state; the
   batched linears over C rows may take a slightly different bf16/HiFi4
   reduction order. Force `hifi4 + fp32_dest_acc_en` (91l:51) on every matmul.
5. **`scatter_row` cost.** 16 full-attn × C × 64 layers of concats. Tiny each;
   if profiling shows dominance, escalate to C'5b (batched SDPA-prefill).
6. **Last-chunk logits handoff.** Off-by-one on slicing the last row corrupts
   the first decode token but does not invalidate the chunk kernel itself.

## 11. Effort estimate

| Task | Hours |
|---|---|
| `deltanet_chunk_ondevice` | 3 |
| Unit test 86 + boundary | 2 |
| 91l wiring + scatter_row + last-chunk logits | 2 |
| Test 87 (real prompt) | 1 |
| 91r `--prefill-mode chunked` | 1 |
| Debug (Q-scaling, conv fencepost, GQA, dtype) | 4 |
| **Total** | **~13h** (≈1.5 days) |

Throughput targets: 5-tok prompt 80 ms; 512-tok ≤ 0.9 s; 32k ~55 s
(vs ~80 min serial).
