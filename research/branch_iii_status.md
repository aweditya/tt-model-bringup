# Branch III — Status

**Target**: Qwen3.6-35B-A3B on 2-chip TP → Qwen3-Coder-Next on full 4-chip quietbox. Local coding/research/learning daily-driver replacement for cloud LLMs.

**Session timestamp**: 2026-05-11 / 2026-05-12

## Phase A — Foundations (DONE except A6 + A7 device-runs)

| Phase | Status | Evidence | Key result |
|---|:---:|---|---|
| **A0** MoE regression diag | ✅ | `research/moe_regression_a0_findings.md` | `mul(silu_g,u)` 137µs (1.7× a unary); topk 100-150µs |
| **A1** Architecture docs | ✅ | `research/qwen36_arch_notes.md` | 40 layers `[L L L F]×10`, all shapes pinned |
| **A2** Equations extracted | ✅ | `research/qwen36_modeling_excerpts.md` | DeltaNet 4-input-proj recurrence + sigmoid gates |
| **A3** DeltaNet isolated | ✅ | `experiments/82_gated_deltanet.py` | **cosine 0.999995**, traced 251µs, 5.3× speedup |
| **A4** Gated Attention isolated | ✅ | `experiments/83_gated_attention.py` | **cosine 0.999943**, eager 592µs |
| **A5** MoE block isolated | ✅ | `experiments/84_moe_block.py` | **cosine 0.9998** (with dev-selected experts), eager 3266µs |
| **A6** Parallel scan | 🟡 v1 script ready | `experiments/85_deltanet_scan_v1.py` | Awaiting qb1 to run |
| **A7** Multi-chip primitives | 🟡 probe script ready | `experiments/87_multichip_primitives.py` | Awaiting qb1 to run |

## Phase B — Integration (PLANNED, not started)

Plan committed at `research/phase_b_integration_plan.md`. 6 substeps, ~15-25 hrs:
- B1 weight load + bf8 quant + per-chip placement
- B2 wire 40-layer decode forward
- B3 correctness gate (per-4-layer cosine + 8/8 token match)
- B4 first real code generation
- B5 trace capture for static layers
- B6 perf measurement

Blocked on A6 v1 + A7 device-runs. Both scripts are written and ready; only need qb1 stable to execute.

## Phases C, D (PLANNED, in branch_iii_kickoff.md)

- **C**: 4-chip TP, bf16, 32K context. ~10-15 hrs.
- **D**: Scale config to Qwen3-Coder-Next (80B/3B active, 512 experts). ~5-10 hrs.

## End-to-end perf math (per A measurements + ceiling analysis)

For Qwen3.6-35B-A3B at decode, single-token forward:

| Component | Per-instance (traced where possible) | Count | Subtotal |
|---|---:|:---:|---:|
| DeltaNet recurrence | 251 µs | × 30 | 7.5 ms |
| Gated Attention SDPA | 400 µs (estimate w/ trace) | × 10 | 4.0 ms |
| MoE block (eager, routing data-dep) | 3.3 ms | × 40 | 132 ms |
| RMSNorm × 2 / layer | ~50 µs each | × 80 | 4.0 ms |
| Misc (residuals, embed) | — | — | ~5 ms |
| **TOTAL single-chip** | | | **~150 ms/tok ≈ 6.5 tok/s** |

With 2-chip parallelism halving the matmul time:
| **TOTAL 2-chip TP eager** | | | **~80 ms/tok ≈ 12 tok/s** |

With expert parallelism for MoE (half of 256 experts on each chip):
| **TOTAL 2-chip TP + expert-parallel** | | | **~50 ms/tok ≈ 20 tok/s** |

**Realistic Phase B target: 15-25 tok/s on 2 chips of qb1.** Usable for daily-driver coding.

## What we learned

### Architectural

- **Gated DeltaNet is implementable in 15 ops on Blackhole**. The "novel SSM/linear-attention" architecture is just elementwise math + reductions when broken down.
- **Decode cost of DeltaNet is independent of context length**. SDPA crosses over to bandwidth-bound at ~256K context; DeltaNet stays cheap forever. This is the architectural reason for 256K context support.
- **bf16 routing drift is real** at 256 experts. We chose a different top-8 subset than fp32, but the model is trained-stable. Don't gate on per-layer cosine alone — use cosine + greedy token match.

### Operational

- **ttnn 0.69 has every primitive we need**, including MoE-specific `all_to_all_dispatch`/`all_to_all_combine`. The plumbing is there; we just need to call it correctly.
- **Two ttnn sharp edges to handle in Phase B**:
  1. `paged_update_cache` requires sharded inputs (production pattern in `demos/generate_moe.py`)
  2. Partial RoPE via slice+concat fails on non-32-aligned dims — need device-side workaround
- **Single-chip utilization invariant matters**: A3 traced at 3.7% of memory ceiling, A4 at <1% (dispatch-bound on short cache), A5 at 1.9%. None of these are bandwidth-bound. Path to higher util is **fusion** of the elementwise sequences, not more chips. Multi-chip is for memory (the model doesn't fit on 1).

### What hurts

- **qb1 SSH flapping** has eaten 2-3 hours of this session. Not a code problem, host operational issue.
- **The Qwen1.5-MoE perf regression (0.68→0.69, -45%) traces to binary mul + topk**. Likely worse for Qwen3.6 with top-8 routing. Phase B accepts the slower baseline; later we can fuse via `ttnn.swiglu` once we figure out its tile-alignment requirements.

## What's running in parallel

| Track | Status |
|---|---|
| **PJRT real-model agent** | Running — porting Qwen2.5-0.5B end-to-end through JAX+PJRT. Will measure tok/s vs native 142 tok/s. Background. |

## Open immediate steps (when qb1 is back)

1. Run `experiments/85_deltanet_scan_v1.py` (A6 v1 chunked-serial). Validate cosine + tokens/sec at T=64, 256, 1024.
2. Run `experiments/87_multichip_primitives.py` (A7). Discover exact mesh API, validate collectives + all-to-all-dispatch.
3. Once A6/A7 are confirmed green, **start Phase B1** (weight loading skeleton).

## Files produced this session

```
research/
  qwen36_arch_notes.md
  qwen36_modeling_excerpts.md
  moe_regression_a0_findings.md
  phase_a3_deltanet_plan.md
  phase_a3_deltanet_results.md
  phase_a4_gated_attention_plan.md
  phase_a4_gated_attention_results.md
  phase_a5_moe_block_results.md
  phase_a6_parallel_scan_plan.md
  phase_a7_multichip_plan.md
  phase_b_integration_plan.md
  branch_iii_status.md            ← THIS FILE
  branch_iii_kickoff.md           ← non-negotiables + master plan

experiments/
  81_moe_regression_micro.py
  81b_swiglu_fusion_probe.py
  81c_swiglu_signature.py
  82_gated_deltanet.py            ← A3 (works on device)
  83_gated_attention.py           ← A4 (works on device)
  84_moe_block.py                 ← A5 (works on device, 256 experts validated)
  85_deltanet_scan_v1.py          ← A6 v1 (ready, not yet run)
  87_multichip_primitives.py      ← A7 (ready, not yet run)
```

Total: ~13 commits, ~3000 lines of research + experiment code added in this session.
