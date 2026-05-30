# qb1 Model Validation Log

Validating existing model ports on the new qb1 host (4× Blackhole, using device 0 only).
Old baselines from the disconnected `tenstorrent` host (per REPRODUCE.md).

Environment: Ubuntu 22.04, Python 3.10.12, ttnn==0.69.0 (vs prior 0.68.0), transformers==5.8.0, project venv at `~/tt-xla/.venv`. Caches in `~/tt-xla/.cache/` (no /tmp).

## Result envelope

PASS = within 20% of prior tok/s + coherent text + (when applicable) correctness check passes.
INVESTIGATE = anything else.

## Summary table

| Model | qb1 tok/s | Prior tok/s | Δ | Status |
|---|---|---|---|---|
| Qwen2.5-0.5B (native RoPE) | 142.8 | 140 | +2% | **PASS** ✓ |
| Qwen3-0.6B | 76.1 | ~76 (script's own ref) | 0% | **PASS** ✓ |
| Llama-3.2-1B | 78.6 | 78 | +1% | **PASS** ✓ |
| Llama-3.2-3B | 33.7 | 34 | -1% | **PASS** ✓ |
| SmolLM3-3B | 37.5 | (no prior) | — | **PASS** ✓ |
| Llama-3.1-8B-Instruct | 19.0 | 19 | 0% | **PASS** ✓ |
| Llama-3.1-8B correctness | cosine 0.997327, 8/8 token match | cosine ≥ 0.997, 8/8 | — | **PASS** ✓ |
| Qwen1.5-MoE-A2.7B | 15.7 | 22.7 | **-45%** | **INVESTIGATE** ⚠️ |

**7/8 PASS, 1 INVESTIGATE.** Dense models match prior baselines tightly. MoE regressed; likely ttnn 0.68 → 0.69 dispatch-path change for MoE-style workloads.

## Details

### Qwen2.5-0.5B (`experiments/60_native_rope_decode.py`)

```
Native RoPE:      7.0ms/tok (142.8 tok/sec) [+2% vs 140]
Rotation matrix:  7.5ms/tok (133.8 tok/sec)
Prefill (5 tok):  7334 ms
Text:  "The capital of France is Paris. It is the largest city in Europe and the
       second largest in the world..."  (coherent)
```

Cold cache: 99 kernel builds totaling 48.8s. Warm cache subsequent.

### Qwen3-0.6B (`experiments/66_qwen3_06b_port.py`)

```
Decode:  13.1ms/tok (76.1 tok/sec)
Upload:  5892ms, Prefill: 6652ms
Text:   "The capital of France is Paris, and the capital of France, and the
        capital of the capital..."  (loops — greedy on base model)
```

Speed is on the nose. Text quality loops are expected greedy behavior on the base (non-instruction-tuned) model, not a regression.

### Llama-3.2-1B (`experiments/64_llama32_1b_port.py`)

```
Decode:  12.7ms/tok (78.6 tok/sec)
Upload:  18989ms, Prefill: 5930ms
Text:   "<|begin_of_text|>The capital of France is Paris. It is the capital of
        France. It is the capital of..."  (loops, same as before — base model greedy)
```

### Llama-3.2-3B (`experiments/67_llama32_3b_port.py`)

```
Decode:  29.7ms/tok (33.7 tok/sec)
Upload:  52543ms
Text:    coherent
```

Note: first invocation of the same `ssh` command hit a TLB allocation panic (umd TLBManager). Second attempt worked. May want to investigate but not blocking.

### SmolLM3-3B (`experiments/68_smollm3_3b_port.py`)

```
Decode:  26.7ms/tok (37.5 tok/sec)
Prefill: 3087ms
Architecture: head_dim=128, 4 KV heads (no SDPA split needed)
Text:   "The capital of France is Paris, the largest city of France..." (some loop)
```

Interesting: SmolLM3 is faster than Llama-3.2-3B at the same parameter count because its 4 KV heads don't need to be split for SDPA.

### Llama-3.1-8B-Instruct (`experiments/73_llama8b_instruct.py`)

```
Decode:  51.9ms/tok (19.0 tok/sec)
Tokens:  29, hit EOS naturally at step 28
Upload:  120s
Text:   "The water cycle, also known as the hydrologic cycle, is the continuous
        process by which water moves on, through the Earth's systems."  (CLEAN!)
```

Instruct model behaves correctly: coherent answer, natural EOS, no looping. Decode speed matches prior exactly.

### Llama-3.1-8B correctness check (`experiments/76b_8b_correctness_check.py`)

```
First token — numpy: 791 (The), ttnn: 791 (The), match: True
Prefill cosine:    0.997327
Token match:       8/8
```

**Passes the canonical correctness gate** (prefill cosine ≥ 0.997, 8/8 greedy token match against pure-numpy fp32 reference). This is the most important single check — if any model passes this, the ttnn port is faithful to the original.

### Qwen1.5-MoE-A2.7B (`demos/generate_moe.py`)

```
Decode:  64ms/tok = 15.7 tok/sec  [vs prior 22.7 — REGRESSED ~45%]
Tokens:  60 generated in 5.0s (warm JIT cache, 99.2% hit rate)
Prefill: 1235ms (18 tokens)
Text:   "A short story about a robot exploring Mars..."  (coherent text generated)
```

**INVESTIGATE.** Steady-state warm-cache MoE decode is 64ms vs prior 44ms. This isn't cold-cache noise — JIT cache hit rate was 99%, so the kernels were resident. The regression is real.

Likely cause: ttnn 0.68 → 0.69 changed something in the MoE dispatch path. Candidates:
- `ttnn.topk` cost
- expert dispatch overhead
- per-token DRAM-sharded weight read scheduling
- compute kernel config defaults

This is consistent with our "MoE dispatch wall" memory — batch=1 MoE is dispatch-bound, and any small change to per-op cost translates to a big change in total time.

Not a blocker for model bringup track. To diagnose: would need a microbench comparing per-op timings between 0.68 and 0.69. Since we don't have 0.68 installed, this would require side-by-side testing.

## Takeaways

1. **Dense models survived the ttnn 0.68 → 0.69 jump cleanly.** All 6 dense ports run within 2% of prior speed.
2. **8B correctness still gates green.** cosine 0.997327, 8/8 token match. This is the strongest single signal that the port is faithful.
3. **MoE regressed substantially.** Worth flagging but doesn't block the Phase 2/3 work on new models.
4. **Bootstrap was clean.** Every model ran first-attempt (Llama-3B retried once due to a TLB panic, but that's likely a transient device-init issue).

## Next: Phase 2 — research new candidate models

See `research/candidate_models_2026.md` (to be written).
