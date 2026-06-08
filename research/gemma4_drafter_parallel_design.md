# Gemma 4 drafter — actual design + the real bug (R-1 finding)

Written 2026-06-08 after reading HF source end-to-end.

## TL;DR

**The drafter is L=1 (as we built). It does NOT need re-bringup at L=K.**

What we got wrong was the **construction of `inputs_embeds` in the
autoregressive K-call loop**. We supplied the wrong "prev" half every
round. Fix the inputs construction and α should jump from ~0 to whatever
the drafter+target distributional agreement actually delivers.

Scope: ~4-6 hours instead of the original 14h re-bringup plan.

## What the drafter actually is

Reading `modeling_gemma4_unified_assistant.py` + `configuration_gemma4_unified_assistant.py`:

- `forward(inputs_embeds, shared_kv_states, attention_mask, position_ids, ...)`
- `inputs_embeds` shape: `[B, L, 2*backbone_hidden=7680]` — supports L≥1
- `shared_kv_states` is one `(K, V)` pair per `layer_type` from target's
  LAST sliding + LAST full attention layer (per-layer KV SHARING, see
  `modeling_gemma4_unified.py:374-446`)
- The drafter has 4 layers. All 4 of them CROSS-ATTEND to `shared_kv_states`
  (they have no own K/V projections — see line 388 of modeling_gemma4_unified.py:
  "Layers sharing kv states don't need any weight matrices")
- Output: `last_hidden_state` (post_projection, [B, L, backbone_hidden]) +
  `logits` ([B, L, vocab])

The mask logic (line 217 of the assistant modeling) says
"There is no difference for the edge case of `q_len == 1` as it acts as
full attention no matter what" — meaning the masks behave naturally at L=1.

This is **not** "parallel K candidates from one forward". The drafter
just supports arbitrary L but is **typically called with L=1** in
production (per HF's own implementation, below).

## How HF actually uses the drafter

`transformers/generation/candidate_generator.py:1276-1410` contains
`AssistedCandidateGeneratorGemma4` (the class for Gemma 4 drafter
candidate generation). The CRITICAL loop is lines 1376-1404:

```python
for _ in range(max_new_tokens):  # max_new_tokens = K
    last_token_embedding = self.target_model_input_embeddings(last_token_id)
    inputs_embeds = torch.cat([last_token_embedding, last_hidden_state], dim=-1)

    outputs = self.assistant_model(
        inputs_embeds=inputs_embeds,
        attention_mask=model_kwargs.get("attention_mask"),
        position_ids=position_ids,
        shared_kv_states=shared_kv_states,
        use_cache=False,
    )

    last_token_id = outputs.logits.argmax(dim=-1)
    last_hidden_state = outputs.last_hidden_state
```

**The `inputs_embeds` is `concat(prev, cur)` where**:
- **prev** = `target_model_input_embeddings(last_token_id)` — TARGET's
  **embed table lookup** on the last predicted token ID. NOT a hidden
  state from anywhere.
- **cur** = `outputs.last_hidden_state` from the PREVIOUS drafter call
  (the drafter's `post_projection` output, shape `[B, 1, backbone_hidden]`)

## What we did wrong

### Wrong #1 — HF oracle construction

Our `experiments/utils/hf_oracle_gemma4_assistant.py:166`:
```python
drafter_inputs_embeds = torch.cat([h_prev, h_last], dim=-1)
```
Where `h_prev` = `target_h[:, -2:-1]` and `h_last` = `target_h[:, -1:]`.

**Wrong**: this uses two CONSECUTIVE target hidden states. The official
construction uses `concat(target_embed(token), prior_drafter_or_target_hidden)`.

### Wrong #2 — scheduler autoregressive chain

Our `spec_dec_scheduler.py:_drafter_autoregressive_K`:
```python
prev_h = target_h_prev_np
cur_h = target_h_last_np
for k in range(self.K):
    inputs_embeds = np.concatenate([prev_h, cur_h], axis=-1)
    out = drf.drafter_forward(state, inputs_embeds, shared_kv)
    candidates.append(int(out["argmax"].flatten()[0]))
    prev_h = cur_h
    cur_h = out["hidden"]
```

**Wrong**:
- prev_h shifts cur_h forward (wrong — should be `target_embed(prediction)`)
- cur_h is right shape-wise (drafter's `hidden` ≈ post_projection output)

### The drafter trace validation

Our v0.4 drafter trace smoke (#255) showed argmax=597 (matches HF). But
that was against our **wrong oracle**. The drafter WEIGHTS are correct;
we just compared against a wrong reference. Re-running against the
corrected oracle will likely still match (it's the same drafter, just
fed the wrong inputs).

## The fix (revised R-2 + R-5 only)

### R-2 (revised): Corrected HF oracle (~2h)

Modify `experiments/utils/hf_oracle_gemma4_assistant.py`:
- Mimic HF's `AssistedCandidateGeneratorGemma4.get_candidates()` loop
- Iterate K times, building `inputs_embeds` from
  `target_embed(last_token_id) + previous_last_hidden_state`
- For round 0: `last_token_id = prompt[-1]`,
  `last_hidden_state = target.hidden_states[-1][:, last_pos:last_pos+1]`
  (which is what HF passes — target's actual hidden at the last position
  given full prefix run)
- Save K-step trajectory: `drafter_argmax_round_{0..K-1}.npy`,
  `drafter_hidden_round_{0..K-1}.npy`, `drafter_inputs_embeds_round_{0..K-1}.npy`

### R-3 (dropped): ttnn drafter unchanged

The existing drafter_forward at L=1 is correct. No changes.

### R-4 (dropped): drafter trace unchanged

The existing trace at L=1 is correct. No changes (it'll just be validated
against the corrected oracle).

### R-5 (revised): scheduler — fix inputs_embeds construction (~2h)

Modify `_drafter_autoregressive_K`:
- Add `target_embed_table` access to scheduler (read from
  `target_state.embed_tt` — read the target embed table on device, build
  numpy version on CPU OR upload the lookup result per call)
- Round 0:
  - `last_token_id = base_token`
  - `last_hidden_state = target_state.last_target_hidden_cur` (already
    stashed by step_forward_v03)
- Round k>0:
  - `last_token_id = previous_drafter_argmax`
  - `last_hidden_state = previous_drafter_hidden` (from `out["hidden"]`)
- `inputs_embeds = concat(target_embed_table[last_token_id], last_hidden_state)`

### R-6 (unchanged): bench (~2h)

Multi-prompt α probe at K∈{3, 5, 7}. Expect α ≥ 0.3 (real spec-dec).

## Open question

The HF candidate_generator slices `last_hidden_state[:, n_last_matches:n_last_matches+1]`
where n_last_matches is the count of accepted candidates from the
previous round. For ROUND 0 of a fresh prompt (n_last_matches=0), this
gets position 0 of the target's hidden states. **But position 0 is the
START of the prompt, not the END.** That doesn't intuitively make sense
for predicting the NEXT token after the prompt's END.

Possible explanations:
- HF generate() internally re-runs target on just the new tokens after
  accept walk, so `model_outputs.hidden_states[-1]` only has the
  new-tokens portion (length 1 or N), not the full prompt
- Or `n_last_matches` semantics are different than I'm interpreting

**To resolve**: run HF assisted_decoding on a canonical prompt with
debug prints inside the candidate generator to see exact shapes. ~1h.

## Updated total scope

| Phase | Scope | Time |
|---|---|---|
| R-1 | Research (DONE) — found the inputs_embeds construction bug | ✅ |
| R-2 | Corrected HF oracle (use HF candidate_generator construction) | 2h |
| R-3 | Dropped — drafter forward at L=1 is correct | 0h |
| R-4 | Dropped — drafter trace is correct | 0h |
| R-5 | Scheduler fix — correct inputs_embeds in `_drafter_autoregressive_K` | 2h |
| R-6 | Bench | 2h |
| **Total** | | **~6h** |

Plus the open question resolution: ~1h.

## Non-negotiables

Same as before — remote-only, permanent files, no /tmp, frequent commits.
