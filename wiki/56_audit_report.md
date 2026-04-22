# Wiki 56: Independent Audit of Performance Numbers

## Scope

Automated audit of all reported performance numbers in PLAN.md, cross-checked against experiment code and wiki entries (exp 81-88).

## Verdict: CLEAN

No errors found. All performance numbers are methodologically sound.

## Verified

- Experiments 82-88 all use correct end-to-end timing (update_buffers + trace + from_dev + argmax)
- PLAN.md clearly labels "Device" vs "E2E" columns
- Wiki 52 honestly documented the timing methodology bug from exp 60-73
- BFP8 MLP numbers (43ms/47ms on 8B) match exp 84 code
- BFP4 failure (0/20 token match) honestly reported

## Warnings (not errors, but caveats)

1. **BFP8 "8/8 match" was thin** — exp 84 used 1 prompt. Fixed by exp 85 (3 diverse prompts, all correct).
2. **Estimated E2E numbers** for Qwen3, Llama-1B, Llama-3B marked with `~` but not directly measured.
3. **Exp 84 used 30 decode steps** — borderline for timing stability (exp 81 used 100).
4. **56.0 vs 56.1ms** rounding discrepancy in PLAN.md (< 0.2%, not material).

## Action Items

- [x] Test BFP8 on diverse prompts (done: exp 85, 3 prompts)
- [ ] Directly measure E2E numbers for 1B and 3B models (currently estimated)
- [ ] Run longer benchmarks (100 steps) for final reported numbers
