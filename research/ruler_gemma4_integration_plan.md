# RULER integration plan (Gemma 4 12B first, 27B/Nemotron-3 as follow-on)

Status: design + scaffold only — no execution this session.
Owner-on-deck: next dev session.
Tracks task #242 (memory `reference_ruler_long_context_benchmark.md`).

## Why

Our ad-hoc needle test has misattributed long-context correctness 3x
(Gemma 4 Round 9 nearly reverted a real bfp8 win; 35B "50% retrieval = coin flip";
Nemotron-3 0/4 — all root-caused to IT-template prompt-shape echo, NOT to model
correctness). RULER replaces that with the synthetic benchmark every paper
cites — single number per (model, task, length), directly comparable to
upstream tables.

## RULER at a glance (from `https://github.com/NVIDIA/RULER`)

- 13 synthetic tasks across 4 categories:
  - Retrieval — `niah_single_1/2/3`, `niah_multikey_1/2/3`, `niah_multivalue`,
    `niah_multiquery`
  - Multi-hop tracing — `vt` (variable tracking)
  - Aggregation — `cwe` (common words), `fwe` (frequent words)
  - QA — `qa_1` (SQuAD), `qa_2` (HotpotQA)
- Synthetic prompts (Paul Graham essays for NIAH haystacks, SQuAD/HotpotQA for QA).
  Prompt SHAPE is hand-authored, not lifted from IT chat templates → does not
  trigger the IT-template-echo failure mode that bites our needle test.
- Lengths 4k / 8k / 16k / 32k / 64k / 128k.
- Driver: `scripts/run.sh <model> <benchmark>` → calls `data/prepare.py` (synth),
  then `pred/call_api.py --server_type {hf|vllm|openai|gemini|trtllm|sglang}`,
  then `eval/evaluate.py` → per-task scalar score.
- `pred/call_api.py` dispatch in `client_wrappers.py`: `OpenAIClient` already
  POSTs `/v1/chat/completions` with `model`, `messages`, `max_tokens`,
  `temperature`, `top_p`, `seed`, `stop` — exactly the shape `cb_api.py` accepts.

## Integration path: fork RULER upstream + drive via `cb_api.py` OpenAI backend

Picked over lm-eval-harness because:

1. RULER ships its own `OpenAIClient` that already speaks the request shape our
   `experiments/serve/cb_api.py` already serves. The lm-eval-harness
   `openai-chat-completions` model adapter exists but adds a second translation
   layer and its scoring assumptions are looser than RULER's per-task metrics
   (which is the whole point of using RULER — comparability with the upstream
   paper table).
2. RULER scoring lives in `scripts/eval/synthetic/constants.py` per-task; we
   want EXACTLY that scoring so our number is comparable to the Gemma / Llama
   / Nemotron paper tables. lm-eval-harness would re-implement and drift.
3. Effort delta: forking RULER's driver = wiring `--server_type openai
   --server_host localhost:8000` (already supported upstream). lm-eval-harness
   would require writing a RULER task config from scratch.

We do NOT vendor RULER into this repo. Plan is:

- Clone `NVIDIA/RULER` to `~/RULER/` on qb2 (one-time, mirrors how we clone
  tt-metal). Pin to a commit SHA in `experiments/bench/ruler/README.md` so
  future runs are reproducible.
- Our wrapper script (`scripts/run_ruler.sh`) sets `OPENAI_API_KEY=dummy` +
  `OPENAI_API_BASE=http://localhost:8000/v1` and dispatches into
  `~/RULER/scripts/run.sh`.
- Our `experiments/bench/ruler/_runner.py` is the thin entrypoint for any
  Python-side glue (chat-template overrides, per-task length sweeps, score
  aggregation). It composes with `experiments/cb/_runner.py` for consistent
  log/format.

## Phase plan

### Phase 1: driver wiring (v0.5.bench.alpha) — ~2 dev hours

- Clone RULER on qb2 → `~/RULER/`, pin SHA in our `README.md`.
- Confirm `OPENAI_API_BASE` env var is honoured by RULER's `OpenAIClient`
  (it is, per OpenAI Python SDK convention). If not, patch + send PR.
- Add Gemma 4 12B entry to `~/RULER/scripts/config_models.sh` with
  `MODEL_FRAMEWORK=openai`, `MODEL_TEMPLATE_TYPE=base` (RULER's "base" template
  bypasses chat templating — our `cb_api.py` server-side `chat_template` is
  what matters, RULER just passes the prompt as one user message).
- Smoke run: `niah_single_1` at L=4096, `num_samples=10` (override). Goal:
  end-to-end pipeline produces a score file, ANY score, no crashes.

### Phase 2: v0.5 acceptance gate task set — ~1 dev hour to define

`experiments/bench/ruler/tasks.yaml` documents the v0.5 cut:

- Tasks: `niah_single_1`, `niah_multikey_2`, `vt`.
- Lengths: 4096, 8192, 16384, 32768.
- `num_samples=50` per task/length (RULER default 500; we cut to keep the
  full sweep under ~1h on Gemma 4 12B at ~50 ms/tok).
- Acceptance: `niah_single_1@4k ≥ 0.90` (must match Gemma upstream within
  noise), no length where any task drops to 0 (sanity floor).

v0.5.bench (full) = all 13 tasks × {4k, 8k, 16k, 32k} × 500 samples;
run once per perf round.

### Phase 3: Gemma 4 12B IT first run (~3 hours wall, after server-up)

Command (documentation stub only — DO NOT run this session):

```bash
# on qb2, with the Gemma 4 12B IT CB server already running on :8000
bash scripts/run_ruler.sh gemma4_12b v0_5_accept
# under the hood:
#   TT_BACKEND=gemma4_12b OPENAI_API_BASE=http://localhost:8000/v1 \
#     bash ~/RULER/scripts/run.sh gemma4_12b synthetic
```

Server-up cost: `experiments/serve/scripts/serve_cb.sh start` ~14 min cold
(Gemma 4 12B bootstrap on qb2). RULER run cost at v0.5 cut: ~3 dev hours
(50 samples × 3 tasks × 4 lengths × ~150 tokens-out × 50 ms/tok = ~4500 s
device-time + per-prompt prefill at 32k = ~0.4 s × 600 prompts = 240 s).

### Phase 4: 27B + Nemotron-3 A/B/C baseline (~1 day total)

Same task set, switching `TT_BACKEND=27b` then `nemotron3_nano`. Three numbers
on the same axes = the model-quality gate going forward.

## Pass/fail target — upstream anchors

We do NOT have a published Gemma 4 12B RULER number yet (model is newer than
public RULER tables we can grep). Use class-similar anchors as ceilings:

| model               | NIAH avg @ 4k | NIAH avg @ 32k | source                                      |
| ------------------- | ------------- | -------------- | ------------------------------------------- |
| Llama-3.1-8B-Instr  | ~95%          | ~78%           | RULER paper table 2                         |
| Llama-3.1-70B-Instr | ~96%          | ~88%           | RULER paper table 2                         |
| Qwen2.5-7B-Instr    | ~93%          | ~63%           | Qwen2.5 tech report                         |
| Mistral-Nemo-12B    | ~95%          | ~84%           | RULER leaderboard                           |
| Gemma-2-9B          | ~92%          | n/a (8k max)   | Gemma 2 tech report                         |
| DeepSeek-V3         | ~96%          | ~90%           | DeepSeek-V3 tech report                     |

Honest framing: we may not match these even with correct serving, because:

- Upstream evaluates BASE models with task-specific sampling; we evaluate IT
  models through our chat template.
- RULER scoring is sensitive to format (e.g. NIAH expects a bare 8-char code;
  our IT model often prepends "The password is ..."). RULER's substring
  scoring tolerates this; exact-match for QA does not.
- Our bf16 chain on Blackhole accumulates ~0.001 drift per layer (memory
  `feedback_bf16_chain_drift_at_B_gt_1.md`); upstream is fp16/bf16 on H100.

Concrete v0.5 acceptance:

- `niah_single_1 @ 4k ≥ 0.85` (within 10% of Llama-3.1-8B class).
- `niah_multikey_2 @ 16k ≥ 0.50` (long-context regression floor).
- `vt @ 8k ≥ 0.30` (variable tracking is harder; this is the "model can
  reason" floor — anything ≥ 0 says it's not hallucinating randomly).

If we miss by a lot, it's a serving bug — fall back to teacher-forced
ladder (memory `feedback_teacher_forced_ladder_method.md`).

## Risks

- **Sliding window vs RULER lengths**: Gemma 4 has 5 sliding-window layers per
  6 with window=4096. NIAH at 32k *must* rely on the 1-of-6 global-attention
  layer for retrieval. This is upstream model behaviour — if we score low at
  32k, that's a model property, not a bug.
- **Compute cost**: 500-sample x 13-task x 6-length full sweep ~24 dev hours
  per model. v0.5 cut keeps us at ~3 hours/model. Don't run full sweep
  speculatively.
- **IT chat-template diff**: our `cb_api.py` applies the Gemma 4 chat template
  server-side via the tokenizer's `apply_chat_template`. RULER passes a
  single user message — fine. The risk is if RULER ever sends a prompt
  containing `<start_of_turn>` etc. that double-renders. Mitigation: keep
  `MODEL_TEMPLATE_TYPE=base` in RULER so it doesn't pre-template.
- **Greedy determinism**: RULER's OpenAI driver passes `temperature=0`. Our CB
  engine maps that to greedy (memory `cb_api.py:_build_sampling`). Good.
- **/v1/chat/completions vs /v1/completions**: RULER uses chat-completions
  (confirmed in `client_wrappers.py`). `cb_api.py` serves both → no issue.

## Known unknowns

- Does upstream RULER's OpenAI client respect `OPENAI_API_BASE`? OpenAI
  Python SDK does, but RULER may construct the client with a hardcoded URL.
  Verify in Phase 1; if not, ~10-line patch + send PR upstream.
- Does RULER's `data/prepare.py` need real HF tokenizer for Gemma 4? Yes —
  tokenizer_path is a required arg. Use `google/gemma-4-12b-it` HF id; qb2
  has HF cache populated.
- 128k length on Gemma 4: model's `max_position_embeddings` is 131072 per the
  config we ship; sliding-window layers handle up to 4k. Expect a steep drop
  at 64k+. OK, document it.

## Next-session entry point

1. Read `experiments/bench/ruler/README.md` for the exact run command.
2. SSH to qb2: `cd ~ && git clone https://github.com/NVIDIA/RULER` (pin SHA).
3. Bring up Gemma 4 12B CB server (`serve_cb.sh start` with
   `TT_BACKEND=gemma4_12b`).
4. Run smoke: `bash scripts/run_ruler.sh gemma4_12b v0_5_accept_smoke`.
5. If smoke passes, run full v0.5 accept set; commit the score JSON to
   `experiments/bench/ruler/results/gemma4_12b/<YYYY-MM-DD>.json`.

Files touched in this scaffold (phase-0 commit):

- `research/ruler_gemma4_integration_plan.md` (this file)
- `experiments/bench/ruler/README.md`
- `experiments/bench/ruler/_runner.py`
- `experiments/bench/ruler/tasks.yaml`
