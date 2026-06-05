# HANDOFF — cold-start one-pager

What this project is, where the perf is now, what to run, and what is next.
Read top to bottom; everything else is linked.

---

## NEW HEADLINE (2026-06-04, post-demo): MM7 — Nemotron-3 Nano 30B-A3B

Stanford CS440LX **demo shipped successfully** 2026-06-04 (27B + Gemma 4 12B
live chat via the TUI, [poster v5](presentation/poster.pdf), Qwen 27B PC
verified end-to-end at 5.1× / 8.0× per-token speedup).

**Next bringup target**: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.
This is a **Mamba2-Transformer HYBRID MoE**, NOT a clean Qwen-MoE clone.

- 52 layers, `MEMEM*EMEMEM*...` pattern → **23 Mamba2 + 23 MoE + 6 Attention**
- 30-31.6B total / ~3.5B active per token
- 6 attention layers have NO RoPE (positional info lives in Mamba2 state)
- MoE is DeepSeek-V3-style (sigmoid + group-restricted top-k + scaling=2.5)
- Experts use `relu²` activation; shared expert is 2× wider than routed
- ssm_state must be fp32 on device

**The blocker**: tt-metal does NOT ship a Mamba2 SSD kernel. 23 of 52
layers depend on it.

**User decision (2026-06-04): Path B — owned Mamba2 SSD kernel up-front.**
G0..G4 staging mirrors the 35B `qwen36_gdn_decode_owned` build. Phase 0
(kernel) before Phase 1 (forward/decode/CB/HTTP ladder). Estimated total
5-8 weeks to v2.

**Plan-of-action**: [`research/nemotron3_nano_30b_a3b_bringup_plan.md`](research/nemotron3_nano_30b_a3b_bringup_plan.md)
**Architecture brief**: [`research/nemotron3_nano_architecture_brief.md`](research/nemotron3_nano_architecture_brief.md)
**Tasks**: #183 (G0 numpy oracle), #184 (G0a harness), #185 (G0b qb1 prep,
parallel), #186 (G1 single-core), #187 (G2 multi-core), #188 (G3 batched),
#189 (G4 server wrapper) — each blocks the next.

**Current work (2026-06-04)**: G0 in progress — Mamba architecture
primer for learning (`wiki/`), then numpy oracle + tt-metal SSM survey.
User explicitly wants to learn the Mamba math themselves.

---

## OLD HEADLINE (kept for reference): demo-day priorities

**Demo-day priorities (2026-06-04 11:55 PT)** — superseded by Nemotron above:
1. **Get the chat TUI rock-solid** — this is the live demo. 27B + Gemma 4 12B are the demo models. Hardening already shipped (commits `c523d28`, `c88f6d5`, `f20bb81`, `0e5a8f5`, `ee7cd20`). Test it live as soon as server is back.
2. **Make sure the server runs perfectly on 27B + 12B**. Existing perf numbers are great (see headline below); just verify nothing broke and screenshots get captured for the poster.
3. **De-prioritise 35B work**. 35B perf/drift/PC fixes can wait — user explicitly said "we can do performance and drift checking for the 35b model haha" meaning skip it.

**Server state**: STOPPED (qb1 in use by a colleague). Restart with the FINAL fast-path config:
```
ssh qb1 'cd ~/tt-xla && rm -f .cache/server_cb.pid && HF_HUB_OFFLINE=1 \
    TT_BACKEND=27b TT_CB_SLOTS=32 TT_CB_PREFIX_CACHE=1 \
    bash experiments/serve/scripts/serve_cb.sh start'
```
For Gemma 4: `TT_BACKEND=gemma4_12b TT_GEMMA4_VARIANT=it ...`. Do NOT pass `TT_CB_TOPK_K` — the default (0 = logits trace or argmax-tail) is what wins.

**Headline perf (HTTP CB, traced, owned_gdn, argmax-tail trace)**:
- Gemma 4 12B IT B=32: **316 tok/s aggregate at 32 clients** (8.35 → 316 = 27.7×)
- Qwen3.6-27B B=32: **232 tok/s aggregate at 32 clients** (8.32 → 232 = 27.9×)
- Single-client streaming: gm4 TTFT 1.40s decode 17.4 tok/s; 27B TTFT 1.70s decode 11.5 tok/s
- Multi-turn HTTP with PC on 27B: turn 2 = 5.99s for 172-tok prompt (PC hit, 6.3× speedup)
- 35B at TT_CB_SLOTS=1: 3.13 tok/s (B>1 blocked by task #162 — won't fix this session)

**TUI is ready** (`scripts/chat.py`, README at `scripts/CHAT_TUI.md`). Key features:
- Claude-Code-style welcome panel (closed box, url + model + cwd + settings)
- `● assistant (<model_short>)` per-turn header with thin grey rule
- **Character-level streaming** — chunks emit to stdout as they arrive
  (no line buffering), so long Qwen3.6 thinks visibly flow
- **Readline editing** in the input prompt — Ctrl-W word-delete, Ctrl-A/E
  line nav, Alt-B/F (= Option-←/→) word nav, ↑/↓ history (persisted
  at `.cache/chat_history`), Ctrl-R reverse search
- `--max 4096` default (up from 1024) so `/continue` is rare
- `<think>…</think>` blocks shown by default (the live token flow is
  the "model is alive" signal); `/think` toggles, `--hide-think` opts
  into the dim "(thinking…)" placeholder at launch
- `/status` (and `/show`) panel; `/clear` (and `/new`) reset; cwd shown as
  `~/…` with path-aware ellipsis for long paths
- `/paste` multi-line mode (terminal-level bracketed-paste also enabled)
- `/yank` copies last assistant reply / code block to system clipboard
- `/metrics [N]` live Prometheus dashboard for N refresh cycles
- `/screenshot` saves to `presentation/screenshots/tui_<ts>.png`
- Expanded shell allow-list (`git`, `grep`, `find`, `python -V/-c`, etc.) with strict deny-list
- `write_file(path, content, mode)` and line-ranged `read_file(path, start=N, n=M)` tools
- Graceful HTTP-error recovery + terminal-state reset on exit

**TUI verified live on 27B 2026-06-04**: banner renders all-four-sides closed,
`<think>` hides cleanly, multi-turn HTTP wall/prompt_t drops from 0.349 → 0.024
across 3 turns (prefix cache hitting; same nickname/fact carry across turns).
Streaming is now char-level (no per-line stall); readline editing (Ctrl-W,
Option-←/→, ↑↓ history) works on both GNU readline and macOS libedit.

**Demo path RE-VERIFIED 2026-06-04 with PC metric inspection**:
- Qwen 27B + TT_CB_PREFIX_CACHE=1, 3-turn coding chat:
  - T0 (36 tok, cold): 8.69 s, wall/prompt_t = 0.241
  - T1 (125 tok, **PC HIT**): 5.84 s, wall/prompt_t = 0.047 (5.1× per-tok speedup)
  - T2 (201 tok, **PC HIT**): 5.92 s, wall/prompt_t = 0.030 (8.0× per-tok speedup)
- Metric delta: pc_hits +2, pc_misses +1, evictions +0 — clean PC story
- Carries the "ADIT" nickname from T0 → T2 chess strategy. Coherent.

**Gemma 4 12B IT also re-verified on TUI 2026-06-04**: streaming works, PC
hits on T1/T2 (wall/prompt_t 0.212 → 0.075). **But** chat output still
duplicates `thought\n` stanzas — the `<|channel>thought\n<channel|>`
chat-template asymmetry (root-caused in `research/gemma4_pc_chat_template_asymmetry_2026-06-04.md`,
task #176).

**Demo plan (user-decided 2026-06-04)**:
- **27B hot** the whole talk — that's the multi-turn / prefix-cache story.
  Demo PC by sending a 2-or-3-turn coding chat; show the wall_s drop on T2.
- **Swap to Gemma 4 12B IT** mid-talk to show insane first-prompt speed
  (~17 tok/s single-client decode, 316 tok/s aggregate at B=32). The
  server restart takes ~1.5 min for Gemma 4 (fast — used as time to talk
  through what's happening on stage). Gemma PC is parked — use cold
  first-prompt only.

**Gemma 4 PC: validator GREEN, live still misses (~80% done, parked)**:

Trail so far (2026-06-04):
- Step 1 ✅ Generic active-prompt suffix detector (commit `184753d`,
  `experiments/serve/openai_endpoint.py:_active_prompt_suffix`).
  Renders the same probe message twice (`add_generation_prompt=True`
  vs `=False`); the divergent tail is the active-only suffix to strip.
  Covers Qwen's `<think>\\n\\n</think>\\n\\n` (5 tokens), Gemma's
  `<|channel>thought\\n<channel|>` (4 tokens), and any future template
  with the same asymmetry. Memoised by `id(tokenizer)`.
- Step 2 ✅ Validator unification (`experiments/cb/validate/pc_token_match.py`):
  imports `_messages_to_prompt` directly so the gate tests the SAME
  code path the live server uses. No more drift between validator and
  prod logic.
- Step 3 ✅ Validator EOS auto-detection: walks the trailing tokens of
  `[user, asst]` rendered with `add_generation_prompt=False` to find the
  chat-end marker. Discovers Gemma's `<turn|>` (id 106) without a
  hard-coded name list; works for any tokenizer.
- Step 4 ✅ Validator result: **3/3 PASS** for both `google/gemma-4-12B-it`
  and `Qwen/Qwen3.6-27B`.

What's left (the ~20%, shovel-ready):
- Step 5 ❌ Live HTTP shows pc_hits=0 / pc_misses=3 / evictions=3.
  Root cause: when the model hits `max_tokens` instead of emitting a
  natural EOS, `cb_scheduler._finish` falls back to
  `next(iter(self.eos_ids))` for the canonical EOS. `self.eos_ids` is
  a `frozenset` — Python's hash-order iteration may land on id 1
  (`<eos>`) for Gemma `{1, 50, 106}`. But the chat template inserts
  id 106 (`<turn|>`) at past-asst boundaries, so the cached
  `prompt + canonical` is one byte off from the next turn's prompt.
- Fix design: add a `chat_end_id` scheduler attribute set at bootstrap
  via the same trailing-token-of-passive-render trick the validator
  uses (so it works for any future backend). Replace the
  `next(iter(eos_ids))` fallback in `_finish` AND `cancel(mark_live=True)`
  with `self.chat_end_id`. Two ~5-line edits, plus a deploy +
  server restart cycle (~6 min for 27B, ~2 min for Gemma 4) to verify
  live. Deprioritised because the demo plan uses Qwen-hot for the
  PC story (Qwen has worked since the 8aeeb53 canonicalise fix on
  2026-06-04 — it picks 151645 as `next(iter({151645}))` from a
  1-element set, which happens to match its chat-end token).

**To run TUI** (once server is back):
```
python3 scripts/chat.py --url http://qb1:8000 --model 'Qwen/Qwen3.6-27B'
# or
python3 scripts/chat.py --url http://qb1:8000 --model 'google/gemma-4-12B' --tools
```

**Open code-only items (parked pending server)**:
- Gemma 4 multi-turn PC: 2nd asymmetry root-caused (`<|channel>thought\n<channel|>` suffix); fix designed in `research/gemma4_pc_chat_template_asymmetry_2026-06-04.md`. Strip-from-cache-only requires scheduler plumbing. PARKED.
- `use_multicore=False` on lm_head argmax (commit `918c025`) — needs server test to confirm perf cost + determinism win.
- TUI screenshots for poster (task #177) — needs server up to demonstrate.

**Poster**: v3 at `presentation/poster.pdf`. Sky-blue Tenstorrent theme, columns rebalanced, both models' streaming numbers in. Compile with `presentation/compile.sh` (lualatex 2-pass) if you edit `poster.tex`.

**Key reference files** (read first if confused):
- `presentation/06_live_measurements.md` — single source of truth for all measurements
- `research/tokenizer_chat_template_reference.md` — universal tokenizer/chat-template gotchas (so we never re-debug this)
- `research/cb_perf_regression_audit_2026-06-04.md` — explains the 13 → 232 tok/s recovery
- `scripts/CHAT_TUI.md` — TUI commands + tools

---

## OLD COMPACTION-READY STATUS (kept for reference)
## 🔥 COMPACTION-READY STATUS (2026-06-04) — read this first

**The user wants you to keep going through the queue below without
asking; they're not stopping for status updates.**

### Live RIGHT NOW
- **Server**: STOPPED (qb1 in use by a colleague, paused at user request).
  Will resume when free. All experiments paused.
- **Code state**: all session fixes committed + pushed (see `git log --oneline -20`).
- **Last verified perf** (HTTP CB, traced, owned_gdn, argmax-tail fast path):
  - Gemma 4 12B IT B=32: **8.35 / 89.25 / 172.52 / 316.12** tok/s at 1/8/16/32 clients (27.7×)
  - Qwen3.6-27B B=32: **8.32 / 61.27 / 117.62 / 232.12** tok/s at 1/8/16/32 clients (27.9×)
  - Single-client streaming: gm4 TTFT 1.40s, decode 17.4 tok/s; 27B TTFT 1.70s, decode 11.5 tok/s
  - 35B at TT_CB_SLOTS=1: 3.13 tok/s (blocked by task #162 for B>1)

### What's queued

| # | Task | State |
|---|---|---|
| 1 | Gemma 4 perf | ✅ +8% vocab-shard, +94% argmax-tail, gm4 = 316 tok/s at B=32 |
| 2 | 27B perf | ✅ +75% cb_dn_recurrence fix, +48% argmax-tail, 27B = 232 tok/s at B=32 |
| 3 | 35B drift / multi-turn Q&A / needle | ✅ resolved + 3/3 PASS + 50/50 (bf16 floor) |
| 4 | Multi-turn HTTP with PC (27B) | ✅ 6.3× speedup on turn 2 (PC hits) |
| 5 | Gemma 4 multi-turn PC | ⏳ scheduler-side canonicalisation shipped (commit `8aeeb53`); validator finds a SECOND chat-template asymmetry (`<|channel>thought\n<channel|>` suffix). Fix designed, parked pending test. See `research/gemma4_pc_chat_template_asymmetry_2026-06-04.md`. |
| 6 | bf16 determinism | ⏳ `use_multicore=False` on lm_head argmax (commit `918c025`); needs server test to confirm perf cost + determinism win |
| 7 | 35B B>1 empty-slot poisoning (#162) | ⏳ unblocks 35B aggregate AND spec-dec |
| 8 | Gemma 4 perf P2 (distributed RMSNorm) | ⏳ design in `research/gemma4_perf_briefing_2026-06-04.md` |
| 9 | Gemma 4 perf P3 (paged SDPA on globals) | ⏳ design in same briefing |
| 10 | TUI hardening | ✅ commits `c523d28`, `c88f6d5`, `f20bb81`, `0e5a8f5`; README `scripts/CHAT_TUI.md` |
| 11 | TUI screenshots for poster | ⏳ needs server back up |
| 12 | Spec-dec (Qwen 3B + 35B) | ⏳ feasibility in `research/speculative_decoding_plan_2026-06-04.md`; blocked on #162 |
| 13 | Cleanup audit (18 items) | ⏳ 3 done (B1 cb_api, A1 docs, determinism); 15 pending |
| 14 | Poster v3 | ✅ Tenstorrent sky-blue, columns rebalanced, streaming numbers in |

### In-flight as of right now (compaction-resilient: re-check these after compaction)
- **35B per-layer drift probe** — `cb35` tmux harness bootstrapping
  (35B = ~14 min bootstrap; ~20/40 layers @ 7 min when last checked).
  Check status: `ssh qb1 'tmux capture-pane -t cb35 -p | tail -10'`.
  Once `[harness] ready. Drop trigger files` appears:
  `ssh qb1 'touch ~/tt-xla/.cache/cb35_runtime/trig/per_layer_drift_pos1'`
  Result: `ssh qb1 'cat ~/tt-xla/.cache/cb35_runtime/trig/last.log'`.
  JSON: `~/tt-xla/.cache/cb35_runtime/per_layer_drift_pos1.json`.
  Pins owned_gdn=ON + dn_state_dtype=bf16 (manual path broken).
  Note: the dev harness file `harness.log` may not get written for cb35
  (open-path quirk under tmux); use `tmux capture-pane` as the live
  source of truth for bootstrap progress.
- **Tracy probe for Gemma 4 perf** — `experiments/utils/tracy_profile_one_gemma4_layer.py`
  shipped (commit `2610ef3`). Run AFTER the 35B drift probe finishes
  and the gm4 harness is back up, to find what's bottlenecking P2/P3.
  Capture cmd in the file's docstring + `research/gemma4_perf_briefing_2026-06-04.md`.
- **No subagents running** — both finished. Briefings saved:
  - `research/gemma4_perf_briefing_2026-06-04.md` — TOP-3 perf opts
    + Tracy capture commands + roofline. Read this first when starting perf.
  - `research/35b_drift_briefing_2026-06-04.md` — drift cliff bisection
    plan (3 steps: P_cliff search → per-layer L_locus → sub-op at
    (L_locus, P_cliff)). Read first when starting 35B drift.
  - The 35B sub-agent confirmed: `server_35b_ttnn.step_forward_inner`
    already writes `capture[f"layer_{L}"]` per layer (no server-side
    diff needed). Just need to write
    `experiments/cb/isolate/cb35_per_layer_drift_pos1.py` (fork
    `gm4_per_layer_drift_pos1.py`, swap to 35B oracle + cb35 harness).
    Use `owned_gdn=ON` ALWAYS — manual path is broken
    ([[feedback-35b-manual-recurrence-path-broken]]).

### What's shipped this session you might lose track of
- **Gemma 4 12B (base + IT)** end-to-end: bootstrap → forward →
  multi-step → long-context (3/3 needle haystack) → trace (3.56×) →
  CB B=4 → HTTP chat. Memory: `[[project-gm4-pos1-cliff]]` (resolved),
  `[[feedback-gemma4-sdpa-scale-1]]`, `[[reference-gm4-dev-harness]]`,
  `[[feedback-harness-state-version-skew]]`.
- **Multi-EOS support** across cb_engine + cb_scheduler + cb_api: reads
  generation_config.json + tokenizer.eos_token_id, accepts list/set.
  Driver: Gemma IT EOS list `[1, 106, 50]`. Smoke result: IT chat now
  has `finish_reason=stop` at 26 tokens (was 80, finish=length).
- **scripts/chat.py rewritten Claude-Code-style**: ANSI panels,
  markdown rendering, slash commands (/save /load /history /tools),
  multi-line input via trailing `\\`, **built-in tool calling** (shell
  /read_file/calc) with `--tools` flag.
- **Model bringup recipe** at `research/model_bringup_recipe.md`
  ([[reference-model-bringup-recipe]]) — the staging ladder, REUSE
  rules, bug catalog, and meta-lesson on why bringup got 10× faster.
  Read FIRST when starting a new model.
- **Allowlist hardened** at `.claude/settings.local.json` — tmux:*,
  env-prefix patterns, deny block for rm -rf / force push / reset
  --hard. (.gitignored)
- **Variant-aware bootstrap** for Gemma 4: `TT_GEMMA4_VARIANT={base, it}`.
  Tokenizer config differs (IT ships a real chat template + eos
  token list); same arch / shapes / weights structure.
- **Dev harness reload extended**: `gm4_dev_harness.py` reloads
  `server_gemma4_*` sibling modules so CB edits land without
  re-bootstrap.

### How to read this back together (90-sec rehydration)
1. `git log --oneline -30` to see this session's commits.
2. Skim `research/model_bringup_recipe.md` (1 page) — what we know about doing bringup fast.
3. Skim `research/gemma4_perf_briefing_2026-06-04.md` (1 page) — the perf attack.
4. Skim `research/35b_drift_briefing_2026-06-04.md` (1 page) — the drift attack.
5. `ssh qb1 'bash experiments/serve/scripts/serve_cb.sh status'` — is the IT server still up?
6. Resume from the queue table above.

### Hard rules (`CLAUDE.md` non-negotiables — restate yourself before each major action)
1. Plan, then act. No hand-wavy claims.
2. Remote-only execution (`ssh qb1` / `qb2`). Never local device code.
3. No `python -c`; no `/tmp`. Permanent files under `experiments/`.
4. Frequent commits.
5. REUSE before write. Every new file cites the existing pattern it forks in the commit message.

---

## Live session state (2026-06-03 — Gemma 4 12B is the active priority)

**Pivot 2026-06-03**: paused 35B drift (#163) and pivoted to Gemma 4 12B
bringup (#165). Driver: 14-min 35B bootstrap per harness restart was
severely rate-limiting iteration AND the harness itself hung silently
mid-investigation (task #166 captures the harden-it-before-next-bootstrap
work). Gemma 4 12B is dense, 12B, dual-attention-type — smaller weights
(~5-7 min bootstrap), structurally interesting (sliding+global), and
exercises position-dependent paths in isolation. If a positional-state
bug lives in our codebase, v0.3 surfaces it without MoE/DN confounders.

### Active — Gemma 4 12B bringup (#165)

- **Plan**: [`research/gemma4_12b_bringup_plan.md`](research/gemma4_12b_bringup_plan.md)
  — start at §"REUSE MANDATE" (always grep for an existing pattern
  before writing new code) → §0 Step 0 pre-flight (DONE) → §2
  code-reuse map at `file:line` → §3 novel items → §4 sub-task
  breakdown with cosine gates.
- **Step 0 DONE 2026-06-03 (commit `4395b28`)**:
  - §6.1 `sliding_window_size` kwarg confirmed on
    `ttnn.transformer.paged_scaled_dot_product_attention_decode`
    — sliding decode is a kwarg flip, not a new kernel. Plan §3.3
    risk closed.
  - §6.3 GELU variant probe (`experiments/utils/gemma4_gelu_variant_probe.py`):
    `ttnn.gelu(x, fast_and_approximate_mode=False)` matches
    `gelu_pytorch_tanh` at cos=0.999998. The fused-activation path
    (`ttnn.mul(.., [UnaryOpType.GELU])`) uses the APPROXIMATE kernel
    — DO NOT mirror 35B's SwiGLU fused pattern in Gemma 4 v0.
  - Bonus: SDPA doc confirms `cur_pos=-1` skips compute for that
    batch slot — may resolve 35B #162 if backported.
- **v0 oracle DONE 2026-06-03**: `experiments/utils/hf_reference_gemma4_12b.py`
  (commit `156dc9f`) produced `.cache/hf_oracle_gemma4_12b/` with
  6-token "The capital of France is" forward; HF predicts " a" at pos 5;
  all 6 L0 sub-captures present (in/post_attn/pre_ff/mlp/post_ff norms
  + mixer_out).
- **v0.1.0 DONE 2026-06-03 commit `b9f3c35`** — bootstrap + embed
  scale + L0 input_layernorm. cos 0.999996 / 0.999991 vs HF oracle.
  Bootstrap **74-84 sec** on qb1 (vs 35B's 14 min — 11× faster).
- **v0.1.1 DONE 2026-06-03 commit `a35525e`** — Q/K/V projections +
  q_norm/k_norm. 7/7 sub-steps PASS at cos ≥ 0.99997. Hit a sharder
  gotcha en route (`ShardTensor2dMesh` vs the correct
  `ShardTensorToMesh(dim=0)`) — one-line fix; memory entry
  `[[ttnn-shard-1d-vs-2d]]`.
- **v0.1.2 DONE** — attention output at pos 0 + o_proj. cos=0.999990
  mad=0.0625. Found via numpy reproducer that Gemma 4 has v_norm =
  `RMSNorm(head_dim, with_scale=False)` applied to V after v_proj
  (memory `[[gemma4-v-norm]]`). 27B/35B don't have this — reading the
  HF source caught what guessing wouldn't. v0.1.2 also surfaced a
  ttnn `all_reduce_async` signature change vs the simple
  `ttnn.all_reduce(cluster_axis=1)` used by 35B (one-line fix in
  `all_reduce_tt`).
- **v0.1.3 DONE 2026-06-03 commit `7f9f396`** — 13/13 sub-steps PASS;
  full L0 forward bit-id to HF at cos ≥ 0.999958. L0 output (vs HF
  hidden_states[1, 0, :]) cos=0.999975. Two-op GELU works per Step 0.2;
  both Gemma 4 post-norms (post_attention + post_feedforward) correct.
- **v0.2 DONE 2026-06-03** — all 48 layers (sliding + global dispatch)
  + final_norm + tied lm_head + 30·tanh logit softcap. Greedy top-1
  matches HF at pos 0: TT argmax=258882 == HF argmax=258882 (`<image|>`).
  final_norm cos=0.999563, logits cos=0.999137. Surfaced two more
  Gemma 4 novelties: per-layer `layer_scalar` ([[gemma4-layer-scalar]])
  multiplied at end of each decoder layer, AND the cosine-is-not-enough
  diagnostic discipline ([[cos-not-enough-also-check-mad]]) — direction
  passed at L0 but magnitude was 18× off, propagating to L1 collapse.
- **v0.3 IN FLIGHT — setup DONE 2026-06-03 commit `cb4e299`**:
  KV caches per layer (sliding [num_blocks, 8, 32, 256] sharded over
  mesh dim=1; global [num_blocks, 1, 32, 512] replicated), SDPA
  program + memory + compute configs (35B B3 recipe — HiFi2 +
  fp32_dest_acc=False per [[fp32-sdpa-cliff-probe]]), RoPE tables
  (sliding theta=10000 full-rotate, global p-RoPE theta=1e6 partial
  0.25). v0.2 probe re-runs with these in place: STILL PASSES.
  Only FORWARD changes remain. Plan §"v0.3 sub-staging" has detailed
  design notes with 35B `file:line` references.
  - v0.3.0 ARGMAX PASS commit `01c88d6` (batch-0 hack)
  - v0.3.0.1 **FULL PASS** commit `e2ae9f2` (2 SDPA calls per
    sliding layer, NKV=1 each, proper GQA). final_norm cos 0.999601,
    logits cos 0.999370, argmax matches HF.
    Memorialized: [[paged-update-cache-nkv-per-chip]],
    [[read-kernel-source-first]], [[use-existing-isolation-probes]].
  - **v0.3.1 FIXED 2026-06-03 commit `c97bf15`** — root cause was
    SDPA `scale=1.0/sqrt(head_dim)` (wrong); Gemma 4 text attention
    sets `self.scaling = 1.0` (HF `modeling_gemma4.py:1178`),
    confirmed in Tenstorrent's in-tree demo (`decode.py:144`). The
    wrong scale was MASKED at pos 0 because a single-token softmax
    is 1.0 regardless of scale. After-fix multi-step: pos 0..5
    cos_final all ≥ 0.997 (was 0.26 at pos 1); 5/6 argmax PASS (pos
    4 cos=0.9984 but argmax differs — bf16 tie noise per
    [[bf16-chain-drift-at-B-gt-1]]). Per-layer drift ladder
    L0-L46 cos > 0.996 at both pos 0 and pos 1. Memory rule:
    [[feedback-gemma4-sdpa-scale-1]].

    The debug ladder remains useful — keep the env knobs
    (GM4_DEBUG_POS, GM4_ROPE_ZERO, GM4_SKIP_SLIDING, GM4_SKIP_GLOBAL)
    and the four isolation probes (gm4_sliding_write_read,
    gm4_global_write_read, gm4_rope_lookup, gm4_per_layer_drift_pos1)
    for future bringup work.

    Full debug writeup in memory `[[project-gm4-pos1-cliff]]` — the
    bisection burned several "masked-at-pos-0" hypotheses (view-decay,
    in-place buffer update, canonical SDPA config) BEFORE landing on
    scale=1.0. None of those earlier fixes moved the 3/6 number, but
    each closed a real Tenstorrent anti-pattern that would have masked
    the real bug, so they ship.
  - **v0.3.2 DONE 2026-06-03 commit `acb20a6`** — 16-token free-run
    coherent: "The capital of France is a city of art, culture, and
    history." End-to-end forward composition validated. Probe:
    `gm4_v032_freerun.py`.
  - **v0.3.3 long-context validation IN FLIGHT** — mirrors 27B/35B's
    needle-haystack + bf16 prefill drift gates.
    - **v0.3.3.a DONE 2026-06-03 commit `a7eef0d`** — per-pos cos
      ladder at L=215 (Wikipedia Eiffel Tower paragraph). All 3
      bf16-aware gates PASS: argmax match 95.81% (≥90%), cos_final
      median 0.9932 (≥0.99), 5th-pct 0.9779 (≥0.95). No cliff.
      Probe: `gm4_v033a_long_cos.py`; oracle:
      `.cache/hf_oracle_gemma4_12b_L215/` (HF needs `.venv-gemma4`).
    - **v0.3.3.c needle-haystack PASS 2026-06-03** — 3/3 Y verdicts
      at L=100/256/512 frac=0.5 (random 8-char passwords retrieved
      verbatim). Eg L=512 needle `FWD7SWFY` → TT generated
      `**FWD7SWFY**`. Probe: `gm4_v033c_needle_haystack.py`. Saved
      under `.cache/needle_haystack_gm4_ttnn/`. Long-context
      retrieval works end-to-end. Decode at ~160 ms/tok eager.
    - **Dev harness for Gemma 4 LIVE 2026-06-03 commit `<gm4-harness>`**
      — `experiments/cb/dev/gm4_dev_harness.py` (forked
      `cb35_dev_harness.py`). Bootstraps Gemma 4 ONCE (~80s), runs
      tests on demand via trigger files; saves ~80s per iteration.
      Launch: `bash scripts/run_harness_tmux.sh gm4`. Run probes via
      `touch tt-xla/.cache/gm4_runtime/trig/<short_name>` (matches any
      probe whose filename ends in `_<short_name>.py`).
    - v0.3.3.b sliding-window invariance at pos > 1024 — pending; not
      blocking (v0.4 trace shipping first).
  - **v0.4 traced decode DONE 2026-06-03 commit `626c67a`** — 100/100
    traced == eager on a non-trivial teacher-forced + free-run sequence
    (`[258882, 236743, 529, 506, 236764, 496, 3207, 529, 1610, 236764]`
    first 10 tokens before the model degenerates to `<image|>`). Eager
    **182.7 ms/tok** → traced **51.3 ms/tok = 3.56× speedup** out of
    the box; trace capture cost 693 ms one-time. Two-phase warmup per
    [[ttnn-multi-trace-two-phase-warmup]]. `trace_region_size=400_000_000`
    on the mesh (default 50 MB OOMs the 48-layer decode trace).
    Validator: `gm4_v04_trace_validate.py`.
  - **v1 CB DONE 2026-06-04** — `server_gemma4_unified_cb.py`
    (forks `server_tp_cb.py`) ships `setup_cb_state`,
    `update_input_buffers_batched`, batched sliding+global layer
    forward (2 SDPA per sliding layer with NKV=1 each), batched
    paged_update_cache + paged SDPA over per-slot KV. **All gates PASS
    at B=1, B=2, B=4**: 3a B=1 == single-slot bit-identical; 3b
    identical-slot bit-identical; 3c distinct-slot with no cross-talk;
    4 slots in B=4 all match their B=1 references. Validators:
    `gm4_v1_0_alloc_smoke.py`, `gm4_v1_4_3a.py`, `gm4_v1_5_3bc.py`,
    `gm4_v1_6_b4.py`. B=4 eager forward ~1.0s/step.
  - **v2 HTTP wire-up DONE 2026-06-04 commit `9a1e45a`** —
    `gemma4_12b` registered in `cb_api.BACKENDS` +
    `cb_scheduler._BACKEND_MODULES`. CB module also exposes
    `cb_reset_slots`, `forward_batch_tp_inner` alias, and `return_topk`
    support to match the scheduler contract. Bootstrap loads the HF
    tokenizer and installs a minimal Gemma chat template
    (`<start_of_turn>{role}\n{content}<end_of_turn>\n`) since the BASE
    model ships no chat template. Logits readback forced to rank-2
    `[B, vocab]` so the scheduler's `t[:B]` works. `curl
    /v1/chat/completions` with `"The capital of France is"` returns
    `"Paris"`. **Known limitation of the BASE model**: `<end_of_turn>`
    is multi-token text (not a special token), so the model emits
    chat-template-echo noise after the answer and runs to `max_tokens`.
    Strip client-side on `<end_of_turn>` until we bring up the
    instruct variant.

  **Gemma 4 12B bringup COMPLETE end-to-end** —
  bootstrap → forward → multi-step decode → long-context → traced
  decode → continuous batching → HTTP chat. Stack is hot on qb1 for
  experimentation (`TT_BACKEND=gemma4_12b bash experiments/serve/scripts/serve_cb.sh
  start`). Per-token decode at **19.5 tok/s single-seq traced**;
  19.5 → projected ~55-65 tok/s aggregate at B=4 traced.

  - **Gemma 4 12B IT (instruct) DONE 2026-06-04 commit `bdd207c`** —
    bootstrap variant-switch via `TT_GEMMA4_VARIANT=it`. Forks the
    base machinery (same arch); the only deltas are weights, the
    shipped chat template, and proper `<end_of_turn>` EOS. v1.6 B=4
    acceptance PASSED at distinct-from-base argmax outputs. End-to-end
    HTTP chat smoke:
        `curl /v1/chat/completions ... "Write a one-sentence summary
         of the city of Paris."`
        → "Paris is a world-renowned cultural and historical capital
           celebrated for its iconic landmarks, rich artistic heritage,
           and sophisticated culinary scene."
    Known followup: IT has three EOS candidates `[1, 106, 50]` in
    `generation_config.json`; cb_engine only gates on one. Extend
    cb_engine to accept a list when that wart matters.

  **Two-variant bringup recipe validated** — the model_bringup_recipe.md
  staging ladder (v0.0 oracle → v1 CB → v2 HTTP) carried us from the
  base model bringup to an instruction-tuned variant in **~2 hours**
  (download + oracle + smoke + EOS fix). The "fork, don't write" rule
  paid out.
- **All computation on (1,4) P150 mesh on qb1**; readback only for
  cosine compare against the HF oracle (matches 27B/35B pattern).
- **Reuse mandate (user-set 2026-06-03)**: every new file must cite the
  existing file it forks (or "no prior art, here's why") in its commit
  message. Deep utility shelf exists: `experiments/cb/_runner.py`,
  `experiments/utils/{ttnn_introspect,hf_reference_35b,cosine_ladder_*,
  test_fused_*_isolated,needle_haystack_*,tracy_*}.py`,
  `experiments/cb/isolate/{paged_sdpa,paged_update_cache,chunked_sdpa,
  owned_gdn,...}.py`, `experiments/serve/{server_35b_ttnn,server_35b_cb,
  server_tp_cb,cb_api,cb_scheduler}.py`. Plan §"REUSE MANDATE" has
  the full table.
- **Step 0 — pre-flight hardware probes (no model upload, ~5 min)**:
  1. **§6.1**: confirm qb1's installed ttnn exposes `sliding_window_size`
     on `ttnn.experimental.paged_scaled_dot_product_attention_decode`.
     If missing, rebuild ttnn or use manual K/V slice fallback before v0.3.
  2. **§6.3**: 1D pointwise check — which ttnn UnaryOp matches
     `torch.nn.functional.gelu(approximate="tanh")` over `[-5, 5]`.
- **v0 staging** (mirrors 35B `research/35b_cb_bringup_plan.md`):
  v0.1 L0-only forward → v0.2 all 48 layers → v0.3 KV cache with
  sliding-window kwarg → v0.4 trace capture → v1 CB B=4 → v2 server
  wire-up + chat smoke.
- **Top NOVEL items** (full detail in plan §3):
  dual head_dim (256 sliding / 512 global), four norms per layer
  (Llama RMSNorm `w`, NOT Qwen `(1+w)` — bit us hard on 35B
  [[qwen36-qnorm-knorm-zero-centered]]), tied embed + sqrt(hidden)
  embed-scale + 30·tanh(x/30) logit softcap, GELU_tanh activation,
  p-RoPE = standard partial RoPE with global head_dim divisor,
  `attention_k_eq_v` on global layers only.

### Parked — 35B drift cliff (#163, #164, #162)

- **#163**: full staging notes in
  [`research/35b_drift_next_session_plan.md`](research/35b_drift_next_session_plan.md)
  §"REAL findings 2026-06-03". Cliff between pos 1 (cos_L32=0.99) and
  pos 5 (cos_L32=0.32); flavor = positional-state bug. Step 1 probe
  wrapper `cb35_drift_cliff_search` deployed but never executed
  (harness restart from hang was killed when we pivoted).
  Cross-pollination: if Gemma 4 v0.3 surfaces a positional-state bug,
  it may share mechanism with this cliff.
- **#164**: manual recurrence path structurally broken
  (cos 0.08 @ pos 0 with owned_gdn=OFF). Orthogonal to cliff;
  fp32 H_t fix (`92b442f`/`8010b3c`) on main routes through broken path.
- **#162**: B>1 batched forward empty-slot poison. Default
  TT_CB_SLOTS=1 for 35B masks it.
- **#166** (NEW): harness hang hardening — line-buffered Python log
  write, 30s heartbeat, top-level try/except. ~12 LOC; ship BEFORE
  next 35B bootstrap.

### Earlier in this session (still active in prod)

- **27B HTTP smoke PASSED** end-to-end on qb1 (commits `97abfab`,
  `a7ea0fe`, `73fd269`). `/v1/chat/completions` returns
  "The capital of France is Paris." in 2.4s.
- **35B HTTP COHERENT at TT_CB_SLOTS=1** (cb_api default for 35B).
  Sample: "Hello" → "Hello! How can I help you today?" with
  `finish_reason=stop`. 35B B>1 broken (see #162 above).
- **35B first inference step crash fixed** in `39f4663` (cb_scheduler
  3-D readback squeeze).

## Project

Qwen3.6-family bringup on Tenstorrent Blackhole (P150 × 4). Production paths:

- **27B dense, 4× P150 TP** — `experiments/serve/server_tp.py` (single-stream
  Unix socket) or via CB (`serve_cb.sh start`, default backend).
- **27B continuous batching** — `experiments/serve/cb_api.py` + `cb_engine.py`,
  served by `experiments/serve/scripts/serve_cb.sh` (the canonical chat path).
- **35B-A3B MoE continuous batching** — same `serve_cb.sh` with
  `TT_BACKEND=35b`. Routes through `server_35b_ttnn.py` (model) + `server_35b_cb.py`
  (batched forward). Production-ready 2026-06-02.

Backend selection: `BACKENDS` registry in `cb_api.py` + identical dispatch
in `cb_scheduler.py`. Adding a new backend = drop a `server_<name>_ttnn.py`
+ `server_<name>_cb.py` + register both in `BACKENDS`/`_BACKEND_MODULES`.

Hosts: `qb1` and `qb2`, both 4× Blackhole P150 with working `FABRIC_1D`.

## Where the perf is now

| Path | Number | Source |
|---|---|---|
| 27B TP single-seq (steady-state, traced) | **12.93 tok/s** (77 ms/tok) | `serve_tp` on qb2 |
| 27B CB B=1 (traced) | 12.96 tok/s (==prod) | `experiments/cb/bench/trace.py` |
| 27B CB B=32 (traced, aggregate) | **150.5 tok/s** (11.6×) | same |
| 27B CB B=64 (traced, aggregate, shift-acc conv1d) | **593 tok/s** (45.8×) | same |
| 35B-A3B traced decode (qb1, after A002+A003+A004+A008+A009) | **81.16 ms/tok = 12.32 tok/s** | `research/35b_perf_milestones.md` |

CB SLO (qb1, 8 clients × 60 s, P5 gate 2026-05-30):
0 errors / 36 requests / 15 tok/s aggregate / **TTFT p99 = 176 ms**
(`experiments/cb/load/concurrent_chat.py`).

## Hardware ceiling

P150 measured: **404 GB/s/chip** DRAM BW, 110 worker cores, 31.81 GB DRAM
(`feedback_p150_memory_bandwidth_measured` in MEMORY.md).
For 35B-A3B with ~3 GB active params/token/chip:
bf16 BW floor ≈ 3.7 ms/tok → **270 tok/s ceiling**;
bf8 BW floor ≈ 1.85 ms/tok → **540 tok/s ceiling**.

The target is the hardware ceiling, not parity with someone else's number.

## Chat path (production)

```bash
# 27B (default backend):
TT_CB_CHUNKED_PREFILL=1 TT_CB_PREFIX_CACHE=1 \
  bash experiments/serve/scripts/serve_cb.sh start   # ~6 min bootstrap; /health → 503 until ready

# 35B-A3B MoE:
TT_BACKEND=35b TT_CB_SLOTS=2 TT_CB_TOPK_K=64 \
  bash experiments/serve/scripts/serve_cb.sh start   # ~6-14 min bootstrap
# verify via:  curl http://qb1:8000/v1/models  (must report Qwen/Qwen3.6-35B-A3B)
# 35B contract: TT_CB_TOPK_K must be >0 (logits-readback broken, #149);
#               cb_api defaults TT_CB_TOPK_K=64 when TT_BACKEND=35b.
bash experiments/serve/scripts/serve_cb.sh status
bash experiments/serve/scripts/serve_cb.sh stop    # SIGTERM → graceful drain → mesh release
```

Env knobs: `TT_CB_PORT=8000`, `TT_CB_SLOTS=4`, `TT_CB_MAX_NEW=1024`,
`TT_CB_MAX_INFLIGHT=64`. Over-cap requests → HTTP 429.

Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`,
`/health`, `/metrics` (Prometheus), `/bootstrap` (stage + elapsed_s).
See README §"Chat server (production)" for `curl` + `openai` client examples.

**Bootstrap observability** (commit `3e150c0`, 2026-06-02). Lifespan
startup blocks uvicorn from accepting HTTP until it yields, so HTTP
health probes can't see the 14-min 35B bootstrap. Three channels:
- `/bootstrap` (JSON): stage + elapsed_s + ready. Only reachable AFTER
  lifespan yields — useful for "is it ready yet" but not for "is it
  stuck during boot".
- `/health` enriched: 503 payload includes `{bootstrap: {…}}` once
  reachable.
- **Side file `~/tt-xla/.cache/server_cb.bootstrap.log`** — appended
  with explicit fsync from the bootstrap thread. Tail-able during the
  lifespan startup phase, when no HTTP endpoint is reachable yet.
  This is the canonical "is it making progress?" probe.

**Current status (2026-06-02 late evening): 35B HTTP server (re)bootstrapping
on qb1**. Background poll task `bz5lcsa9n` is watching for ready. v0/v1/v2
device primitives + cb35_prod_topk all PASS via the dev harness; the HTTP
wrap is what's flaky to bring up (uvicorn lifespan + 14-min boot + worker-
thread stdout buffering). Once the side-file shows `[harness ready]`, the
chat TUI / curl should work end-to-end.

## Deploy hygiene before any `serve_cb.sh start`

Always sync the entire `experiments/serve/*.py` before starting the server.
The dev harness uses `importlib.reload` per test so it only needs the file
under test; the production server boots a fresh process and reads qb1's
filesystem ONCE.
- One MM1 commit (`418f9cc`) sat in local git but never reached qb1; the
  server kept loading 27B for hours of debugging until we caught it.
- See memory `[[deploy-serve-files-too]]`.
- Quick command: `bash scripts/deploy.sh experiments/serve/*.py`

## What's next

**Prefix caching — END-TO-END VALIDATED (2026-06-01).** Slot-level
content-keyed prefix cache for the CB scheduler. Returning chats reclaim
their live slot at `cur_pos = len(matched_prefix)`, skipping re-prefill of
the history. qb1 smoke test:
- Turn 1 (cold, miss): **5.33s**
- Turn 2 (warm, **HIT**): **2.73s** — 1.96× speedup, 1 cache hit, 0 misses
- Qualitative: turn 2 correctly continued conversation (France → Germany / Berlin)

**Recommended runtime config (2026-06-02):** `TT_CB_CHUNKED_PREFILL=1 TT_CB_PREFIX_CACHE=1`.
CW1 fix (`ea9aa20`) makes both flags coexist by skipping the eager
chunked-prefill fallback for L > chunk_size — the L > chunk_size path
takes the legacy 1-tok/iter route via the decode trace, which is
allocation-free and safe alongside captured traces. Cold-start L > 32
prompts pay 1-tok/iter latency (~80 ms/tok), but with prefix caching
catching turn-2+ this is rare in chat workloads. Verified on qb1 smoke
2026-06-02: 1.97× turn-2 speedup preserved, no wedge.

Gated by `TT_CB_PREFIX_CACHE=0/1` (default 0). `TT_CB_PREFIX_TTL_S=300` for
stale-slot cleanup. Metrics in `/metrics`: `cb_prefix_cache_{hits,misses,
evictions}_total`, `cb_prefix_cache_live_slots`, `cb_prefix_cache_enabled`.

Load-bearing fixes from the smoke debug chain:
- `mark_live` fires on `max_tokens` cap (not just EOS) — `4acc955`
- `tokens_so_far` keeps trailing EOS for chat-template compat — `184f00e`
- `_messages_to_prompt`: `preserve_thinking=True` + trailing-only `<think>` strip — `2cad663`

Plan + per-milestone status: [`research/27b_prefix_caching_plan.md`](research/27b_prefix_caching_plan.md).
Research:
- [`research/vllm_prefix_caching_audit.md`](research/vllm_prefix_caching_audit.md) — APC design
- [`research/vllm_chat_template_handling.md`](research/vllm_chat_template_handling.md) — Qwen3.6 quirks + upstream-blessed `preserve_thinking` fix

Regression gate: `experiments/cb/isolate/chat_template_invariant.py`
(7 cases including 239/239 long-prompt, unicode, 3-turn compound).
Memory: [[prefix-caching-design]], [[qwen36-preserve-thinking]].

**S2 — chunked prefill — LIVE in production (2026-06-01).** CB serves with
`TT_CB_CHUNKED_PREFILL=1`: traced chunked prefill at chunk_size=32 for L ≤ 32,
legacy 1-tok/iter fallback for L > 32. Two-phase warmup (compile-all-then-capture-all)
solves the multi-trace coexistence wedge per [vLLM #352](https://github.com/tenstorrent/vllm/issues/352).
Plan + post-mortem: [`research/27b_prefill_trace_plan.md`](research/27b_prefill_trace_plan.md).

Deferred / superseded: T3 multi-chunk traced prefill (chat win comes from skipping
re-prefill via prefix caching, not making re-prefill faster). Bigger chunk_size
(same reasoning). Both revisitable for long single-prompt cases (no prior cache
to match) after prefix caching ships.

## Roadmap (priority order, 2026-06-02)

1. **DONE — `chunked_prefill=1 + prefix_cache=1` coexist** (CW1, commit `ea9aa20`).
   Run prod with both flags on. Long L > chunk_size prompts use 1-tok/iter
   fallback (slow but allocator-safe). Validated 2026-06-02: 1.97× turn-2
   speedup preserved + no wedge.
2. **DONE — `TT_BACKEND` env selector** (MM1, commit `418f9cc`). cb_api.py
   has a `BACKENDS` registry; `TT_BACKEND={27b,35b}` switches at boot.
3. **IN PROGRESS — 35B-A3B CB bringup (CB35-v0..v4)**. Plan:
   [`research/35b_cb_bringup_plan.md`](research/35b_cb_bringup_plan.md).
   Research basis: [`research/tt_metal_moe_cb_patterns.md`](research/tt_metal_moe_cb_patterns.md)
   (DeepSeek-V3 + Llama-70B-Galaxy patterns).
   - **v0 BIT-VALIDATED 2026-06-02** (commits `112d72a`, `49778b3`):
     `cb35_v0_smoke` 3/3 + `cb35_v0_chat` (8-tok decode bit-identical).
     Critical fix: `base.reset_caches_ttnn` leaks per-layer tensors;
     `cb_reset_states` explicit-deallocs first.
     `return_logits=True` raises (35B [1,VOCAB] readback broken, #149).
     cb_engine routes 35B through topk via `TT_CB_TOPK_K=64`.
   - **v1.0 alloc + v1.1 embed/RoPE BIT-VALIDATED 2026-06-02** (commits
     `ec01052`, `cf211a4`). `setup_cb_state(B>1)` allocates B-leading
     `cb_dn`/`cb_kv` + per-slot `cb_page_table_tt`. `_batched_prelude`
     reads `cb_tok_buf` / `cb_rot_idxs_buf`. Gate: 3/3 alloc + 4/4 prelude.
   - **v1.2 batched DN BIT-VALIDATED 2026-06-02** (commit `4546d29`).
     `dn_step_batched_35b` cos=1.0/mad=0.0 at B=1 and B=2 slot 0. Rank-3
     `[B, N, D]` rms_norm fix ([[ttnn-rms-norm-shape-drift]]).
   - **v1.3 batched GatedAttention BIT-VALIDATED 2026-06-02** (commit
     `e8f2d82`). `attn_step_batched_35b` cos=1.0/mad=0.0 at B=1 and B=2
     slot 0 — first-try pass. Rank-3 `_apply_partial_rope_b` sidesteps
     base's K-broadcast workaround. Paged KV write + SDPA decode use
     `cb_cur_pos_buf`/`cb_page_table_tt` for per-slot context.
     `setup_cb_paged_cfgs` builds B-sized HEIGHT_SHARDED L1 mem cfg +
     SDPA progcfg (B=1 reuses base's `state.paged_*`).
   - **v1.4b broadcast MoE BIT-VALIDATED 2026-06-02** (commit `a6ac640`).
     True broadcast Pattern A — `[E_LOCAL, B, HIDDEN]` middle-dim bump
     in batched expert matmul. mad=0.000000 vs base at B=2 slot 0.
     Fixed the 13% per-slot drift from v1.4 loop. Critical bug in init
     port: used `MOE_INTER_CHIP (128)` instead of `MOE_INTER (512)`.
   - **v2 trace capture SHIPPED 2026-06-02** (commit `c547419`).
     Two-phase warmup + `begin_trace_capture`/`end_trace_capture`
     around `forward_batch_tp_inner` works at B=2. Replay 149.7 ms/step
     vs eager 296.7 ms/step = **1.98× speedup**. cb_scheduler trace
     plumbing inherits automatically (calls the unified entry).
     Higher B → larger speedups.
   - **CB35-prod wire-up GATED 2026-06-02** (commit `f1f7a61`).
     Unified `forward_batch_tp_inner` dispatches B=1→base (v0 bit-id) /
     B>1→v1 batched. Now supports `return_topk=K` at both Bs. cb_scheduler
     can drop in without changes. `cb35_prod_topk.py` 4/4 PASS:
     B=1 top-1 = 8 (matches v0), B=2 distinct prompts produce distinct
     top-1 tokens. Ready for cb_api/cb_engine end-to-end at
     `TT_CB_SLOTS=2`.
   - **v1.5 full B>1 forward FUNCTIONAL PASS 2026-06-02** (commit `0a50e97`).
     `forward_batch_tp_inner_batched` + `layer_forward_batched_35b` —
     40-layer chain at B>1 runs end-to-end. v1_chat results:
     - ✓ slot 0 != slot 1 with distinct prompts (per-slot independence)
     - ✓ slot 0 == slot 1 with same prompt (determinism)
     - ⚠ slot 0 != B=1 ref by argmax tokens — bf16 chain precision drift
       compounding across 40 layers (every individual op verified bit-id
       at B=2 slot 0; the chain noise is what flips argmax tokens). See
       [[bf16-chain-drift-at-B-gt-1]] for the lesson. **Production-shippable** —
       each slot's generation is valid; use cosine for benches, not
       exact tokens.
   - v1 (true B>1 batched forward): next. ~3-5 days.
   - v2 (trace capture at B=N): ~1-2 days.
   - v3 (owned-GDN batched FOLD trick): optional.
   - v4 (prefix cache for attn layers only): LOW PRIORITY (vllm#36493 reports
     ~0.1% hit rate on this arch class — DN layers can't be cached).
   - **Dev iteration harness** (`experiments/cb/dev/cb35_dev_harness.py`):
     bootstrap-once long-running python on qb1, watches `~/tt-xla/.cache/cb35_runtime/trig/`,
     reloads via `importlib`. Cuts fix-test cycle from ~14 min to seconds.
     **MUST launch via `bash scripts/run_harness_tmux.sh`** — nohup + disown
     and setsid + double-fork both fail to keep the python alive after SSH
     disconnect on qb1 (it dies before completing bootstrap). tmux
     survives by design. See `research/35b_cb_bringup_plan.md` for usage.
4. **Multi-model fleet** — plan: [`research/multi_model_serving_plan.md`](research/multi_model_serving_plan.md).
   Once 35B is live, MM5 (Mistral Small 3.2 24B) is the strongest
   framework-generalization test (different vendor, different tokenizer,
   pure dense GQA, no DN). Candidate research:
   [`research/home_llm_landscape_2026.md`](research/home_llm_landscape_2026.md).
5. **Long-context concurrent stress test (MM3)** — validate PC works at
   L=1000+ prompts under realistic concurrency. Reuses
   `experiments/cb/load/concurrent_chat.py`. Final-validation step after
   the multi-model work is in place.

**35B perf** (parallel track). Next levers tracked in
[`research/35b_perf_milestones.md`](research/35b_perf_milestones.md):
async all_reduce overlap, expert-broadcast elimination, routing-weight
fusion, bf8 expert weights.

## Repo entry points

- README — install + demos.
- CONTRIBUTING — dev loop, canary gate, code style.
- `research/` — design docs + living plans (index: `research/README.md`).
- `wiki/` — Q&A wiki, learning-by-building.
- `models/` — multi-model demos (Llama, Qwen2.5, SmolLM, 8B).

## Read order when resuming work

1. This file.
2. [`research/profiling-quick-reference.md`](research/profiling-quick-reference.md) — Tracy + tt-perf-report capture/analyze.
3. [`research/35b_perf_milestones.md`](research/35b_perf_milestones.md) — 35B perf trajectory.
4. [`research/27b_cb_scope.md`](research/27b_cb_scope.md) — CB design + numbers (CB0–CB4).
5. [`research/35b_tt_perf_report_findings.md`](research/35b_tt_perf_report_findings.md) — empirical writeup behind 35B advice.

## Load-bearing rules (each cost a multi-day debug)

- **View-decay**: `ttnn.slice` / `ttnn.reshape` return views. Never
  `ttnn.deallocate` the source while a view is live; clone when in doubt.
- **+1 zero-centered RMSNorm offset** on `q_norm` / `k_norm` /
  `input_layernorm` / `post_attention_layernorm` / `final_norm` (Qwen3.6).
- **K-broadcast RoPE workaround** in the SDPA path — sidesteps a ttnn
  `[1, HEAD_DIM]` slice/concat bug.
- **bf16 KV cache** required by paged SDPA (fp32 hard-rejected).
- **HiFi4 + `fp32_dest_acc_en`** on every matmul (the 91f recipe); mixing
  fidelities corrupts ops silently on Blackhole.

## Workflow

- Profile-driven only. Cite a Tracy / tt-perf-report number for any
  optimization claim. Frame deltas as Δ from BW floor.
- Correctness gate: 5-token Paris (`"The capital of France is" → " Paris..."`)
  on prefill IDs `[2614, 314, 279, 369, 11751]`.
- Iterations in git history or `scratch/`, never in demo scripts.
- Remote-only execution (`ssh qb1` / `ssh qb2`); no device code locally.
