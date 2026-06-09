# Gemma 4 #289 Step 2: selective `fp32_dest_acc=False` plan (2026-06-09)

Step 1a (sliding QKV fuse) is correctness-verified — long-context
argmax gate green at all four L (commits `c7d35e4` + `c1a3c3a`). Step 1b
parked (gelu fold blocker).

Step 2 is the bigger projected win from #292 research:
**selectively disable `fp32_dest_acc_en` on small-K matmuls while
keeping it on the long-context-load-bearing ones**.

## The math (recap from #292 research)

- BF16 dot-product accumulation error: O(sqrt(K) · eps_bf16)
- Invisible to cosine for K ≤ ~4096
- Matters for o_proj (K = num_q_heads × head_dim), down_proj (K =
  intermediate_size), lm_head (K = hidden, feeds argmax), and
  attention's S·V where K = sequence length
- DeepSeek-V3 explicitly keeps embeddings, output head, MoE gate,
  norms, and attention operators in BF16/FP32 — strongest signal in
  the literature

## Gemma 4 12B matmul inventory + safety ranking

| Matmul | K dim | Per-fwd count (full 48-layer) | Safety |
|---|---|---|---|
| **q_proj** (sliding) | 3840 | 40 (or 1 if fused via #293) | SAFE — disable |
| **k_proj** (sliding) | 3840 | 40 (or 1 if fused) | SAFE |
| **v_proj** (sliding) | 3840 | 40 (or 1 if fused) | SAFE |
| **q_proj** (global) | 3840 | 8 | SAFE |
| **k_proj** (global) | 3840 | 8 | SAFE |
| **gate_proj** | 3840 | 48 | SAFE |
| **up_proj** | 3840 | 48 | SAFE |
| `o_proj` (sliding) | 4096 (16 × 256) | 40 | **KEEP fp32_acc** — borderline K, conservative |
| `o_proj` (global) | 8192 (16 × 512) | 8 | **KEEP fp32_acc** — large K |
| `down_proj` | 15360 | 48 | **KEEP fp32_acc** — large K |
| `lm_head` | 3840 | 1 | **KEEP fp32_acc** — feeds argmax, magnitude matters |
| paged_sdpa (S·V) | seq_len | 48 (per layer) | **KEEP** — embedded in kernel, can't toggle separately |

**Disable-safe count per fwd**: 40+40+40+8+8+48+48 = **232 matmuls/fwd**
(or with QKV fuse: 1+8+8+48+48 = 113, but each fused matmul
accumulates Q+K+V together so the typecast saving still applies).

## Implementation strategy

### Approach: per-call compute kernel config selector

Don't make `HIFI4` config a constant. Make a helper:

```python
HIFI4_FP32_ACC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,    # default — load-bearing path
    packer_l1_acc=False,
)

HIFI4_BF16_ACC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=False,   # safe on small-K
    packer_l1_acc=False,
)
```

Update each matmul site to pick one or the other based on the
**safety ranking above** (NOT the env gate — the env gate is for the
bisect probe, not for production runtime).

### Bisect-by-op probe before flipping

Before changing 232 matmul call sites in one commit, do a per-op probe:

`experiments/cb/isolate/gemma4_fp32_acc_bisect_probe.py`:
- env gates per op (e.g. `GM4_BF16_ACC_Q_PROJ=1`)
- one knob at a time
- runs the long-context argmax gate after flipping each knob
- output: dict of {op: pass/fail} so we know which K-bucket is the
  cliff (if any)

This finds the worst-case K beyond which fp32_acc is load-bearing,
not just the per-our-list ranking.

### Long-context gate threshold

Argmax gate at L = {128, 512, 1024, 2048} is the existing baseline.
But the worry per #292 research is the **K = sequence_length** in
attention's S·V — which our gate exercises up to L=2048 only. For full
confidence we should also include L = 4096 in the gate (close to
MAX_KV=4096 in our config) and ideally L = 8k or 32k once chunked
prefill (#290) makes those tractable.

**Decision**: extend `gemma4_long_context_argmax_gate.py` to L=4096
BEFORE Step 2 ships. (L=4096 sequential prefill costs ~190s — slow
but bearable for a one-time baseline; trivial during --verify replay.)

## Phased rollout — SIMPLIFIED 2026-06-09

The per-op bisect (phases 2c → 2d → 2e flipping one op family at a
time) is **overkill given the research already gives the safe contract**.
Decision: flip all 5 op families (Q, K, V, gate, up) together behind
ONE master env gate `TT_GM4_BF16_ACC_SMALL_K=1` and validate
once via --verify @ 5 L's. If green → ship. If fail → bisect.

| Phase | What | Status |
|---|---|---|
| 2a | HIFI4_BF16_ACC config + `_small_k_matmul_config` helper; flip Q/K/V (sliding + global) + gate/up (DRAM + non-DRAM) sites behind TT_GM4_BF16_ACC_SMALL_K | ✅ shipped `689db4f` |
| 2b | Extend `argmax_gate` to L=4096; re-capture baseline | ✅ code shipped `689db4f`; baseline capture **in flight** |
| 2c | --verify with TT_GM4_BF16_ACC_SMALL_K=1 at 5 L's | pending baseline |
| 2d | Tracy delta — total Typecast count drop | pending verify |
| 2e | Needle haystack @ L=4k (secondary precision gate) | pending verify |
| 2f | Flip default; document safety contract | pending all gates |

## Original per-op bisect plan kept here for reference if 2c fails

If the single-flip --verify fails at any L, fall back to bisect:
- (i) Flip Q proj only → --verify
- (ii) Flip K + V proj → --verify
- (iii) Flip gate + up → --verify
First op-family that fails IS the precision cliff; KEEP that one
enabled and ship the rest.

## Expected impact

- Typecast in safe matmuls accounts for somewhere between 5-10% and
  20% of total device kernel time (we'll know after Step 1a tracy
  delta gives us a per-matmul-class breakdown)
- Plus Step 1a saves another ~5%
- Combined projected: 47 → ~32 ms/tok (1.5×), matching #292's research
  projection

## Risks

- **Borderline K bucket**: if K=4096 is actually load-bearing despite
  the research note's "K ≤ 4096 is safe" claim, we lose precision at
  L=2048+ in argmax_gate. Mitigation: bisect probe catches it before
  multi-op flip.
- **Cosine vs argmax sensitivity**: argmax gate is integer-coarse;
  precision drift could push a top-2 token to top-1 only at very long
  context. Mitigation: needle haystack as the secondary gate at the
  end of Step 2g.
- **Interaction with #293 QKV fuse**: the fused qkv_proj_combined
  matmul also wants `fp32_acc=False`. Since the fuse is bit-equivalent
  to three separate matmuls (which would each individually be SAFE),
  the fused version is also SAFE. Just need to remember to flip the
  fused weight's call site too.

## Non-negotiables (sticky)

- Permanent files only — bisect probe under `experiments/cb/isolate/`
- Remote-only Python — runs via ssh
- No /tmp — outputs to `.cache/perf_logs/`
- Frequent commits — one per phase (2a → 2h)
- Plan first → execute → verify — argmax gate is the contract
