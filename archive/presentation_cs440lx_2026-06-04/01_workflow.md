# Workflow: how three model bringups got progressively faster

Bringing up a new LLM family on Tenstorrent Blackhole (Qwen3.6-27B dense, Qwen3.6-35B-A3B MoE, Gemma 4 12B hybrid) takes weeks the first time and ~36 hours the third time, despite each model being architecturally new. The speedup is not from cleverer kernels — it is from a methodology that turns every bringup into a *forking* exercise on top of stable infrastructure. The canonical recipe lives in `research/model_bringup_recipe.md`; every clause below is something the recipe enforces.

## The staging ladder is non-negotiable

Every model walks the same ladder. The gate on each rung is a cosine threshold against an oracle, not "it produces text."

| Stage | Adds | Gate |
|---|---|---|
| **v0.0** | HF oracle: CPU bf16 forward with `output_hidden_states=True` + forward hooks, saved to `.cache/hf_oracle_<model>/` | Oracle dir exists; layer count matches config |
| **v0.1** | (1,4) mesh bootstrap + weights upload + L0 input_layernorm only | cos ≥ 0.999 on `embed_scaled` + `in_norm` |
| **v0.1.x** | One L0 sub-op at a time (q/k/v proj → q/k/v norm → RoPE → SDPA → o_proj → MLP) | cos ≥ 0.999 per sub-op |
| **v0.2** | All N layers + final_norm + lm_head + softcap | argmax matches HF at pos 0 |
| **v0.3** | Paged KV cache + multi-step decode | argmax matches HF for tokens 0..7 |
| **v0.3.3** | Long context (cos ladder at L≥200, needle haystack) | argmax match ≥ 90%; median cos ≥ 0.99 |
| **v0.4** | Trace capture (two-phase warmup) | 100 traced steps == 100 eager, token-for-token |
| **v1** | Continuous batching at B=4 (3a/3b/3c gates) | B=1 bit-identical to single-slot; identical-slot match; distinct-slot isolation |
| **v2** | HTTP wire-up (`cb_api.BACKENDS` + `cb_scheduler._BACKEND_MODULES`) | `curl /v1/chat/completions` returns sensible text |

**Gemma 4 12B walked this entire ladder from oracle → HTTP chat in ~36 hours.** That includes finding three model-specific bugs (SDPA `scale=1.0` instead of `1/sqrt(d_k)`, per-layer `layer_scalar` buffer, `v_norm` with `with_scale=False`) and a paged-cache shape contract (`paged_update_cache` treats `input.dim(1)` as batch — see commits `e2ae9f2`, `c97bf15`). The 35B-A3B took ~2 weeks; the 27B took ~3.

## HF oracle as ground truth, not "compare to a TT path that mostly works"

`experiments/utils/hf_reference_<model>.py` runs HF on CPU once, dumping `hidden_states.npy [N+1, seq, H]`, `logits.npy`, `argmax.npy`, plus per-sub-op L0 captures via forward hooks. Every subsequent TT change is gated against this snapshot via cosine + argmax — never against "the previous working TT path," which conflates a regression with a numerics improvement. The pattern is rigid enough that `hf_reference_gemma4_12b.py` is a direct fork of `hf_reference_35b.py` with two hook lists swapped and the model-structure walk made tolerant of HF's `model.language_model.layers` vs `model.model.layers`.

## Isolation probes before full forwards

`experiments/cb/isolate/` holds one probe per primitive: `paged_sdpa.py`, `paged_update_cache.py`, `dn_recurrence.py`, `chunked_sdpa.py`, plus per-model probes (`gm4_sliding_write_read.py`, `gm4_per_layer_drift_pos1.py`, etc.). The rule is enforced in memory as `[[use-existing-isolation-probes]]`: **before iterating on a kernel call in a 75s+ full forward, grep `experiments/cb/isolate/` for an existing probe and fork it.** A probe round-trip is seconds; a full-forward iteration is minutes plus a 5-14 minute bootstrap. Gemma 4 v0.3.0 burned ~10 minutes on shape-contract failures before this rule was followed; v0.3.0.1 shipped in one harness cycle once the probe surfaced the NKV-per-chip contract.

## The dev harness amortizes the bootstrap

`experiments/cb/dev/gm4_dev_harness.py` and `cb35_dev_harness.py` are long-lived tmux'd Python processes that bootstrap the model once and accept tests via the file system:

```bash
# one-time on qb1 (eats 80s for 12B, ~14 min for 35B)
bash scripts/run_harness_tmux.sh gm4

# per iteration (locally) — runs in seconds
bash scripts/deploy.sh experiments/serve/server_gemma4_unified_ttnn.py \
                      experiments/cb/isolate/gm4_v033a_long_cos.py
ssh qb1 'touch tt-xla/.cache/gm4_runtime/trig/v033a_long_cos'
ssh qb1 'cat tt-xla/.cache/gm4_runtime/trig/last.log'
```

The harness loop polls a trigger directory, `importlib.reload`s the probe module on every fire, and exposes `_reload` to refresh the server module without restart. Probes follow a single contract: `def main(state=None)` so they run standalone or against the resident harness state. Test cost drops from one full bootstrap per iteration to ~30 sec, which is what made 35B drift work tractable at all (per-iter cost dropped from ~14 min to ~30 sec). The harness has its own hardening story — silent stdout backpressure under tmux required dropping `tee`, adding a 30 sec heartbeat, and wrapping the loop in a top-level try/except (commit `84efe50`).

## REUSE mandate: cite the prior art in the commit message

Every new file or function must cite the file it forks (or "no prior art, here's why") in its commit message. The 27B and 35B bringups produced a deep utility shelf — `experiments/cb/_runner.py`, the `cosine_ladder_*` family, `needle_haystack_*`, `test_fused_*`, the entire `experiments/serve/server_*.py` stack — and the cost of building that shelf has already been paid. Gemma 4's bringup plan (`research/gemma4_12b_bringup_plan.md`) opens with a 24-row reuse table mapping each Gemma 4 concern to the existing file that handles it. The decision rule is mechanical: dict registries get edited in place; tokenizer-driven helpers get reused as-is; model-specific shapes get forked.

## Living plans in checked-in docs

Multi-step initiatives live as markdown under `research/<model>_bringup_plan.md`, updated as stages complete. The plans store: goal, success criteria, ordered stages with status, current blocker, links to commits and probes. The `gemma4_12b_bringup_plan.md` table shows v0.1.0 through v1.6 with the gate result and commit SHA on each row. This makes Claude context compaction a non-event — a fresh session reads the plan + recent commits and resumes mid-stage without re-deriving strategy.

## Two-phase trace warmup

`ttnn.begin_trace_capture` cannot tolerate JIT compilation between captures: a decode warmup that compiles ops the prefill didn't will allocate kernel-cache buffers on top of prefill's reserved trace memory, and replay reads garbage. The fix (documented upstream in `tenstorrent/vllm#352`) is two-phase: compile every path with `enable_trace=False` first, synchronize, then capture all traces back-to-back. Without this, Blackhole hangs at 99% CPU on the second capture. `TT_METAL_TRACE_ALLOC_TRACKING=1` names the offending op before corruption occurs.

## Why bringup got 10× faster

Three things compound. **Forking, not writing**: each model copies 80-90% of the previous one's CB module, dev harness, probes, and oracle script. **The bug catalog is finite**: after three bringups the same ~12 footguns reappear (`ttnn.slice` view-decay, RMSNorm `(1+w)` vs Llama-style `w`, `paged_update_cache` NKV contract, `ShardTensorToMesh` vs `ShardTensor2dMesh`) and are recognized in seconds. **The validation infrastructure is generic**: HF oracle, cosine ladder, needle haystack, per-layer drift ladder — none care which model you are bringing up, so the gates write themselves. Concretely: the first 27B perf wins (vocab-sharded lm_head, paged SDPA) each took weeks; the same vocab-shard fork on Gemma 4 landed in a single day for +8% (51.3 → 47.5 ms/tok, 21 tok/s, 100/100 token-for-token vs eager — commit cited in `feedback_p22_gm4_vocab_shard_result.md`).
