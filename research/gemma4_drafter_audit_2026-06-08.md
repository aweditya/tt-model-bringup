# Gemma 4 drafter spec-dec — line-by-line HF source audit (2026-06-08)

Triggered after R-6c (EMBED_SCALE fix) delivered 5× α uplift but rounds 3-4
of the drafter chain still drift from HF. User asked: "is there anything
else we're missing in the spec decoding logic that would fix the remaining
drift? worth doing another research/scraping audit to make sure we're
doing everything we should be."

Scope: all sources read from qb2's `.venv/.../transformers/` end-to-end:
- `generation/candidate_generator.py:1276-1410` — `AssistedCandidateGeneratorGemma4`
- `models/gemma4_unified_assistant/modeling_gemma4_unified_assistant.py` — drafter
- `models/gemma4_unified/modeling_gemma4_unified.py` — target backbone + shared attention class

## TL;DR — what we found

| # | Item | Status | Severity |
|---|---|---|---|
| 1 | `inputs_embeds = concat(target_embed(token), last_hidden)` ordering | ✅ matches HF | — |
| 2 | EMBED_SCALE = sqrt(HIDDEN) ≈ 61.97 on target embed lookup | ✅ FIXED R-6c | — |
| 3 | `position_ids = [[L-1]]` constant across K rounds (NOT incremented) | ⚠️ we don't pass it | **HIGH suspect for chain drift** |
| 4 | **Drafter applies RoPE on Q at position L-1** | ❌ **we SKIP RoPE entirely** | **HIGH** |
| 5 | `outputs.last_hidden_state` returned = `post_projection(model_norm_out)` | ✅ matches HF | — |
| 6 | `outputs.logits` computed from RAW `model_norm_out` (NOT post_proj) | ✅ matches HF | — |
| 7 | `use_ordered_embeddings` / masked_embedding path | ✅ FALSE for 12B; not needed | — |
| 8 | `shared_kv_states` sliced to `current_length` per call | ✅ we do (`L_kv=L`) | — |
| 9 | bf16 chain hidden state precision (post_projection round-trip) | ⚠️ each round bf16→fp32→bf16 | MEDIUM (compounds) |
| 10 | `attention_mask` HF `model_kwargs.get("attention_mask")` — usually all-ones or None | ✅ we don't construct; SDPA `is_causal=False` | LOW |
| 11 | Drafter `pre_projection` Linear(2H, H) before backbone | ✅ matches HF | — |
| 12 | Drafter `q_norm` (no `+1.0` offset, Llama-style) | ✅ matches HF (`with_scale=False`-equivalent in our impl) | — |
| 13 | `lm_head` no softcap (drafter config `final_logit_softcapping=null`) | ✅ matches HF | — |
| 14 | SDPA `scale=1.0` (Gemma 4 sets `self.scaling=1.0`) | ✅ matches HF | — |

**Conclusion**: The highest-suspect remaining bug is #4 — drafter Q has NO
RoPE applied. HF applies it at `position_ids=[L-1]`. The fact that our
chain probe Variant C achieves 3/5 match (rounds 0-2 OK, rounds 3-4 fail)
suggests the drafter is empirically robust to wrong-position Q-RoPE in
early rounds (hidden state dominates), but the wrong attention pattern
compounds with bf16 chain drift to flip argmax at rounds 3-4.

## HF call chain (verbatim from source)

### `AssistedCandidateGeneratorGemma4.get_candidates()` — lines 1330-1410

```python
# Top-of-call setup (line 1366):
current_length = input_ids.shape[1]
shared_kv_states = {
    k: (v[0][:, :, :current_length, :], v[1][:, :, :current_length, :])
    for k, v in shared_kv_states.items()
}
last_hidden_state = last_hidden_state[:, n_last_matches : n_last_matches + 1]
last_token_id = input_ids[:, -1:]
position_ids = torch.tensor([[input_ids.shape[1] - 1]],   # ← CONSTANT for the whole K-loop
                              dtype=torch.long, device=...)

# Per-round loop (lines 1376-1404):
for _ in range(max_new_tokens):  # = K
    last_token_embedding = self.target_model_input_embeddings(last_token_id)  # ScaledWordEmbedding
    inputs_embeds = torch.cat([last_token_embedding, last_hidden_state], dim=-1)
    with torch.no_grad():
        outputs = self.assistant_model(
            inputs_embeds=inputs_embeds,
            attention_mask=model_kwargs.get("attention_mask"),
            position_ids=position_ids,                  # ← SAME EVERY ROUND
            shared_kv_states=shared_kv_states,
            use_cache=False,
        )
    last_token_id = outputs.logits.argmax(dim=-1)
    last_hidden_state = outputs.last_hidden_state       # = post_projection(model_norm_out)
```

**Key observation**: `position_ids` is computed ONCE before the loop and
NEVER incremented. The drafter sees the same position L-1 for all K rounds.

### `Gemma4UnifiedAssistantForCausalLM.forward()` — lines 143-211

```python
inputs_embeds = self.pre_projection(inputs_embeds)         # Linear(2H, H)
bidirectional_masks = self.create_attention_masks(...)
outputs = self.model(                                       # = Gemma4UnifiedTextModel (target backbone)
    inputs_embeds=inputs_embeds,
    attention_mask=bidirectional_masks,
    position_ids=position_ids,
    shared_kv_states=shared_kv_states,
    use_cache=False,
)
last_hidden_state = outputs.last_hidden_state               # = model.norm(...) output
projected_state = self.post_projection(last_hidden_state)   # Linear(H, BACKBONE_H)
logits = self.lm_head(last_hidden_state)                    # ← RAW, NOT projected
return Gemma4UnifiedAssistantOutput(
    last_hidden_state=projected_state.to(source_device),    # ← what we feed back
    logits=logits.to(source_device),
)
```

### `Gemma4UnifiedTextModel.forward()` — lines 620-690

```python
hidden_states = inputs_embeds
position_embeddings = {}
for layer_type in self.unique_layer_types:
    position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

for decoder_layer in self.layers[:num_hidden_layers]:
    hidden_states = decoder_layer(
        hidden_states,
        shared_kv_states=shared_kv_states,
        position_embeddings=position_embeddings[layer_type],   # ← cos, sin for position L-1
        ...
    )

hidden_states = self.norm(hidden_states)                       # final RMSNorm
```

### `Gemma4UnifiedTextAttention.forward()` shared-kv branch — lines 405-440

```python
cos, sin = position_embeddings                                 # = RoPE table at position L-1
query_states = self.q_proj(hidden_states).view(hidden_shape)
query_states = self.q_norm(query_states)
query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)   # ← !!! RoPE on Q !!!
query_states = query_states.transpose(1, 2)

if self.is_kv_shared_layer:
    key_states, value_states = shared_kv_states[self.layer_type]   # already RoPE'd by target
    key_states = key_states.to(query_states.device)
    value_states = value_states.to(query_states.device)
```

**Q gets RoPE at position L-1 unconditionally.** K comes pre-RoPE'd from
the target's actual KV cache.

## Our impl gaps

### Gap #4 (HIGH) — RoPE skipped on Q

`server_gemma4_12b_assistant_ttnn.py:587, 629` — comment reads "RoPE
skipped — position_ids=[0] → cos=1, sin=0 → identity." But HF passes
`position_ids = [[L-1]]`, NOT [[0]]. We're missing the rotation.

For a 5-token prompt: HF Q has been rotated by angle `θ_d × 4` per dim;
ours is unrotated. K (from target) has been rotated at positions
`[0..4]`. Attention scores `q·k_i` capture relative position; ours
misreports it as `-i` instead of `(L-1)-i = 4-i`.

The drafter's hidden-state input (concat of `embed(token)`, `prev_hidden`)
already encodes position info because `prev_hidden` is the target's hidden
at the current generation position. This explains why our chain probe
Variant C round 0 still matches HF even though Q's RoPE is wrong — the
drafter has learned to extract enough position info from `prev_hidden`
to argmax correctly when the chain is fresh.

Once `prev_hidden` becomes our drift-noisy version (rounds 3-4 in chain
probe), the missing RoPE on Q stops being masked by the hidden-state
position signal, and argmax flips.

### Gap #9 (MEDIUM) — bf16 chain on `out["hidden"]`

`drafter_forward` line 770 does:
```python
hidden_tt = ttnn.matmul(final, state.post_projection_tt, compute_kernel_config=HIFI4)
hidden_np = _readback_replicated(hidden_tt, state.mesh).reshape(B, L, BACKBONE_HIDDEN)
```

Then scheduler `_drafter_autoregressive_K` uploads `last_hidden` (the
returned `hidden_np`) as bf16 next round. Each round: bf16 → fp32 → bf16
roundtrip + 4-layer backbone bf16 precision compounds.

HF computes the whole chain in bf16 without ever going to fp32. The
post_projection output stays as bf16 tensor throughout.

We can mitigate by:
- (a) keeping `out["hidden"]` as a device tensor between rounds (no
  readback) — requires scheduler change
- (b) running `post_projection` with fp32 accumulator (already HIFI4)
- (c) accepting bf16 chain noise as fundamental

## Proposed fix order (high → low leverage)

### F-1 (HIGH leverage, MEDIUM scope): Add RoPE on Q at position L-1

Plan:
1. Add `cur_pos: int` parameter to `drf.drafter_forward` (default = 0
   for back-compat). Scheduler passes `cur_pos = L_prompt - 1` (constant
   across all K rounds, matching HF).
2. Inside `_drafter_attn_sliding` / `_drafter_attn_full`, after `q_norm`
   and before SDPA permute, apply RoPE on Q at `cur_pos`. The drafter's
   RoPE base / dim split matches the target's per-layer-type rotary_emb
   — reuse target's cos/sin tables. (Target server already has these
   cached: `state.rope_cos_full`, `state.rope_sin_full`, similar for
   sliding.)
3. Update isolation probe `gemma4_assistant_pre_projection_smoke.py`
   shape sanity.
4. Re-run chain probe `gemma4_spec_dec_drafter_chain_probe.py` —
   Variant C should rise from 3/5 → 5/5 if hypothesis holds.
5. Re-run multi-prompt smoke — mean α should rise from 0.067.

Risk: Q dim 0 ranges depend on whether Gemma 4 uses NEOX-style
"half-half split" or "interleaved pairs" RoPE. Match the target's existing
convention in `server_gemma4_unified_ttnn.py` (the K/V was RoPE'd that
way during target's prefill, so Q must use the same).

Estimated time: 2h (1h impl + 30m probe + 30m smoke).

### F-2 (MEDIUM leverage, MEDIUM scope): Keep hidden on-device

Plan:
1. Modify `drafter_forward` to ALSO return `hidden_tt` (the device
   tensor) alongside `hidden_np`.
2. Scheduler `_drafter_autoregressive_K` keeps `hidden_tt` across rounds
   instead of readback + re-upload. Round 0 uploads `target_h_last`
   normally; subsequent rounds reuse the device tensor.
3. The `embed(token)` half still needs upload per round, but it's a
   single row lookup.
4. Concat is `ttnn.concat([embed_tt, hidden_tt], dim=-1)` on device.

Risk: device memory budget; need to deallocate prior round's hidden.

Estimated time: 3h.

### F-3 (LOW leverage): fp32 accumulator on post_projection

Already at HIFI4 — minimal gain. Skip.

## Non-negotiables

- Remote-only: all reads on qb2 via `ssh qb2`; no local Python invoked.
- Permanent files: this doc, no inline scripts.
- No /tmp: probes go in `experiments/cb/isolate/`.
- Frequent commits: one per logical step (audit doc, fix-1, fix-2, etc.).
- Reuse mandate: F-1 forks target's existing rotary_emb tables; F-2
  forks the on-device tensor flow from server_gemma4_unified_ttnn.

## Decision

Proceed with **F-1 (RoPE at position L-1)** as the next action — highest
leverage, well-scoped, fully diagnosable via existing chain probe.

If F-1 closes Variant C to 5/5 → F-2 only matters for perf (cuts ~1ms
per round, K=5 → 5ms saved) and can be deferred.

If F-1 keeps Variant C at 3/5 → escalate to F-2 + per-round hidden
cosine probe instrumentation to localize.
