# Post-Correctness Roadmap (Parking Lot)

Once Qwen3.6-27B generates coherent text on Blackhole (i.e., `'Paris'`
appears in the prefix prediction), the following items become active.
Items are ordered by quality-impact first, then performance.

## Quality / robustness

- **Quality re-check at 100+ tokens**: confirm the model stays coherent past short prompts
- **bf16 residual ablation**: B'9 promoted x to fp32 to test a (wrong) drift hypothesis. With the real bugs (#1-5) fixed, bf16 residual is probably fine. Test:
  - Revert 91l's embed upload to bf16, cos/sin to bf16, A_log/dt_bias to bf16, conv_state to bf16
  - Re-run 91l 60 tokens
  - If quality survives → keep bf16 (16% faster, half the activation memory)
  - If quality degrades → keep fp32 OR identify which channel actually needs fp32 (likely just the residual stream, not all the other tensors)
- **Numerical stability**: softplus via `log(exp(x)+1)` overflows for large x; switch to torch-style softplus (`max(0, x) + log1p(exp(-|x|))`). Likely small effect.
- **Long-context generation**: verify quality holds past 500 tokens (KV cache fills, DeltaNet state accumulates more state-update steps)

## Process / sanity

- **Wiki entry: `bringup_checklist.md`** with the mandatory pre-flight steps:
  - [ ] Enumerate all safetensors keys per layer
  - [ ] Cross-check vs loader (we now have `experiments/utils/weight_audit.py --diff-loader`)
  - [ ] Validate ONE layer's output against an external oracle (HF, vLLM, etc.), NOT against your own numpy
  - [ ] Audit every RMSNorm formula (`w*x` vs `(1+w)*x`)
  - [ ] Audit every gating activation (silu vs sigmoid)
  - [ ] Audit every GQA repeat (tile vs interleave — `experiments/utils/repeat_semantics_probe.py`)
- **Per-substep numpy reference** in `experiments/utils/hf_layer0_substep_dump.py` is now the canonical oracle path. Update B'2/B'3 numpy ref to match HF substep-by-substep.

## Performance (after correctness AND quality lock)

- **B'6.5 — paged_update_cache**: pad n_kv to 32 OR build custom sharded layout to unblock. Eliminates the per-layer numpy roundtrip (~16 MB/token PCIe).
- **Trace capture**: enable `ttnn.begin_trace_capture`/`execute_trace` to eliminate dispatch overhead. Expected 4-5× speedup (per exp 95).
- **Native RoPE**: `ttnn.experimental.rotary_embedding` is 2.6× faster than the rotation matrix we use (see `feedback_native_rope.md`).
- **Per-position prefill**: feed all prompt tokens in one chunk via `chunk_gated_delta_rule` instead of sequentially. Faster prefill, same quality (hopefully).
- **bf8 lm_head + GPTQ**: lm_head is 1.3 GB of weights. Quantizing it (currently bf8) further or replacing with INT4-GPTQ could save 1 GB.
- **Performance run**: full 60-token generation, measure tok/s, compare against bf16 ablation, against trace, against B'6.5 fix. Document the wins.

## Multi-chip (post-single-chip-baseline)

- **Phase A7 / multi-chip TP**: now unblocked on qb2 (fabric works). Could enable batched serving across 4 chips.

## Documentation debt

- `research/b9_final.md` (write after generation works): summarize the 5 bugs found, the validation methodology that emerged, the wins.
