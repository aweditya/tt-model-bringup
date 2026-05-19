# Reorg + Rename Plan — 2026-05-19

Research-only proposal. No files moved/renamed/deleted by this document. The
user reviews, then a separate execution agent applies the changes.

Inputs surveyed: top-level dirs (`experiments/`, `research/`, `wiki/`,
`pjrt_plugin/`, `demos/`, `home/`, `tt_docs_corpus/`, root files); 181
`experiments/*.py` files (mostly numbered `NN_*.py` + `91*` Qwen3.6-27B chain);
148 `experiments/utils/*.py` probes; `experiments/owned_ops/` (9 Qwen3.6-27B
GDN/conv1d kernels); `experiments/serve/` (server_tp, server, clients); 120
`research/*.md` files; existing `research/qwen3_coder_30b_a3b_plan.md` (so a
30B-A3B target is already on file); `.git/config` remote
`https://github.com/aweditya/tt-xla.git`; Claude settings dir
`/Users/adityasriram/.claude/projects/-Users-adityasriram-Labs-stanford-cs440lx-tt-xla/`
(exists, 185 MB, two JSONL session logs + 135-file `memory/` dir).

---

## 1. Reorg — per-model isolation

### Proposed top-level layout

```
<repo_root>/
  CLAUDE.md, HANDOFF.md, PLAN.md, REPRODUCE.md, requirements.txt   (kept at root)
  models/
    qwen36-27b/                  # current production target
      experiments/               # 91*, demo_qwen36_27b.py, jax_qwen05b_*
      owned_ops/                 # 9 qwen36_* kernels + chain test
      serve/                     # server_tp.py, client_tp.py, scripts/serve_tp.sh
      research/                  # qwen36_arch_notes, deltanet_*, gdn_*, qb2_*,
                                 #   c4_*, c7*, b8/b9, mtp_*, paged_sdpa_*,
                                 #   needle_haystack_*, owned_*, owned_gdn_*,
                                 #   integration_*, qb1_*, fabric_diagnosis,
                                 #   qb2_decode_profile_2026_05_15.md, etc.
      probes/                    # utils/{attn,deltanet,mlp,conv1d,rope,
                                 #   sdpa,kv,paged,tp_,p1..p25,mtp_*}_*.py
    qwen36-30b-a3b/              # next bring-up
      research/                  # seed with qwen3_coder_30b_a3b_plan.md +
                                 #   models_landscape_2026 excerpt
      experiments/  serve/  owned_ops/  probes/   (empty scaffolds for now)
  shared/
    serve/                       # protocol.py, generic _tp_all_reduce, mesh helpers,
                                 # bench harness, RPC utils extracted from server_tp.py
    probes/                      # mesh_smoke_probe.py, fabric_*, kernel_profile_probe,
                                 # cumsum_probe, neumann_inverse_probe, slice_rope_probe,
                                 # rope_scaling_probe, hf_download.py, npz_inspect.py
    research/                    # blackhole_hardware_tricks, jax_infrastructure_*,
                                 # multi_chip_*, p150 roofline, profiling_guide,
                                 # advanced_optimization_techniques, performance_ceiling*,
                                 # memory_audit_2026-05-14, friend_repo_*, models_landscape,
                                 # candidate_models, gemma4_arch, kimi_glm_bringup_menu,
                                 # 01-06 numbered hardware/JAX notes, ACTIVE_CONTEXT.md
    wiki/                        # entire current wiki/ (curriculum, not model-specific)
    tt_docs_corpus/              # scraped tt docs
  pjrt_plugin/                   # untouched (orthogonal JAX/XLA arm)
  demos/                         # tutorial demos, GPT-2 / Llama / general — kept as-is
  legacy/                        # 01..90 numbered scratch files that are
                                 # neither model-specific nor reusable;
                                 # alternatively leave in shared/probes/legacy/
  .cache/  .claude/  home/  tenstorrent  (untouched)
```

### Per-asset routing rules

- `experiments/91*_qwen36_27b_*.py`, `experiments/demo_qwen36_27b.py`,
  `experiments/90_qwen36_weight_skeleton.py` -> `models/qwen36-27b/experiments/`.
- `experiments/owned_ops/qwen36_*` -> `models/qwen36-27b/owned_ops/`.
- `experiments/serve/server_tp.py`, `client_tp.py`, `scripts/serve_tp.sh`,
  drift scripts -> `models/qwen36-27b/serve/`. **Split before move:**
  factor `_tp_all_reduce`, mesh open/close, paged-SDPA wrapper, RPC framing,
  bench plumbing into `shared/serve/tp_runtime.py`; leave the Qwen3.6-27B
  model graph (`gated_attn_step_tp`, `deltanet_step_tp`, vocab-sharded
  lm_head, on-device embed/cos/sin) in the model dir.
- `experiments/serve/server.py`, `client.py`, `scripts/serve.sh` — qb1
  single-chip server. Currently Qwen3.6-27B-only -> `models/qwen36-27b/serve/`
  with the same `shared/serve/` factoring (paged SDPA wrapper, bench harness).
- `experiments/serve/protocol.py`, `tests/test_protocol_mock.py` -> `shared/serve/`.
- `experiments/utils/*.py` (148 probes): tag-route by filename prefix.
  Qwen3.6-27B-specific (`attn_*`, `deltanet_*`, `mlp_tp_*`, `conv1d_*`,
  `paged_sdpa_*` at prod shape, `tp_*`, `p1..p25_*`, `mtp_*`, `needle_*`,
  `cosine_ladder_*`, `gdn_*`, `owned_conv1d_*`, `qb2_gdn_*`,
  `qb2_tp_*`, `rope_variants_*`, `qwen_chat_template_*`,
  `bf8_kv_cache_*`, `fp32_kv_cache_*`, `gqa_prerepeat_*`) ->
  `models/qwen36-27b/probes/`. Generic
  (`mesh_smoke_*`, `fabric_*`, `kernel_profile_*`, `cumsum_*`,
  `neumann_inverse_*`, `slice_rope_*`, `rope_scaling_*`, `hf_download.py`,
  `npz_inspect.py`, `analyze_tracy_overlap.py`) -> `shared/probes/`.
- `experiments/01..89_*.py` (numbered curriculum + GPT-2/8B/MoE bring-up):
  not Qwen3.6-27B, but is the learning trail. Leave under `legacy/` or
  `shared/probes/legacy/`. Do NOT delete — `HANDOFF.md` and wiki entries
  cite by filename.
- `research/*.md`: split per the inline table above. Files named
  `qwen36_*`, `deltanet_*`, `gdn_*`, `owned_*`, `qb1_*`, `qb2_*`, `b8/b9_*`,
  `c0..c7_*`, `mtp_*`, `paged_*`, `needle_*`, `kv_cache_*`,
  `integration_*`, `phase_a*/phase_b*`, `closing_performance_gap`,
  `per_layer_cosine_milestone`, `moe_*` -> `models/qwen36-27b/research/`.
  Files named `01..06_*`, `blackhole_*`, `jax_*`, `multi_chip_*`,
  `models_landscape_*`, `candidate_models_*`, `gemma4_*`,
  `performance_ceiling*`, `memory_audit_*`, `friend_repo_*`,
  `model_bringup_plan`, `mixed_precision_strategies`, `ACTIVE_CONTEXT.md`,
  `profiling_guide` -> `shared/research/`.
- `research/qwen3_coder_30b_a3b_plan.md` -> `models/qwen36-30b-a3b/research/`
  (seed the new model dir with it).
- `wiki/`, `tt_docs_corpus/`, `pjrt_plugin/`, `demos/`, `home/`, root `demo.py`,
  `tenstorrent` (text dump), `PLAN.md`, `REPRODUCE.md`: untouched on first pass.

### Execution hygiene

- One `git mv` per file/dir (preserves blame). Do it in batches by routing
  rule, one commit per batch.
- After each batch, grep for hardcoded paths
  (`experiments/serve/server_tp.py`, `experiments/91f_*`, `experiments/utils/...`)
  in `HANDOFF.md`, `research/ACTIVE_CONTEXT.md`, every `.md`, and every
  `scripts/*.sh`. Update references in the same commit.
- Add `models/qwen36-27b/README.md` and `models/qwen36-30b-a3b/README.md`
  pointing at the canonical entry script and current target tok/s.
- Update `CLAUDE.md` "Project Structure" block to match.
- The qb1/qb2 rsync targets (`~/tt-xla/`) are independent of local layout
  but the per-host script paths in `experiments/serve/scripts/*.sh` and
  `HANDOFF.md` §4 (server lifecycle) WILL break — update in the same PR.

---

## 2. Local folder rename

### Candidate names (descriptive, no jokes)

1. `tt-blackhole-llm` — covers the actual scope (Tenstorrent Blackhole + LLM
   bring-up), still short.
2. `qwen-blackhole` — model-and-hardware framing; reads well in a path.
3. `tt-metal-models` — flags the real underlying stack (tt-metal, not XLA).
4. `tt-bringup` — generic, future-proof if more models land.
5. `blackhole-decode` — emphasises the optimisation focus (decode tok/s).

Recommendation: **`tt-blackhole-llm`** (descriptive, unique, survives adding
30B-A3B / GLM / Kimi).

### Steps (do NOT execute now)

Assume the chosen name is `tt-blackhole-llm`. Old: `tt-xla`.

1. Quit Claude Code on this repo. Confirm no other process holds the cwd.
2. `mv /Users/adityasriram/Labs/stanford/cs440lx/tt-xla
      /Users/adityasriram/Labs/stanford/cs440lx/tt-blackhole-llm`.
3. Rename the Claude settings dir. The current dir
   `/Users/adityasriram/.claude/projects/-Users-adityasriram-Labs-stanford-cs440lx-tt-xla/`
   exists (185 MB, contains two session JSONLs + `memory/` with 135 files).
   The folder name is the slug Claude derives from the cwd at session start.
   Rename to match the new path:
   `mv /Users/adityasriram/.claude/projects/-Users-adityasriram-Labs-stanford-cs440lx-tt-xla
       /Users/adityasriram/.claude/projects/-Users-adityasriram-Labs-stanford-cs440lx-tt-blackhole-llm`.
   The JSONL session files inside are content-addressed by UUID; they keep
   working. `memory/MEMORY.md` references files by basename, not path, so
   nothing inside needs editing.
4. Open Claude Code in the new path. Verify `memory/` and session history
   are picked up (the slug match is what binds them).
5. Update local references: any shell aliases, VS Code workspace,
   `~/.zshrc` / `~/.bashrc`, `~/.ssh/config` `RemoteCommand`, browser
   bookmarks pointing at `file:///.../tt-xla/`, and `.envrc` if present.
6. Inside the repo, update `CLAUDE.md` title, `HANDOFF.md` intro, and any
   absolute-path examples (most are remote-relative `~/tt-xla/`, see below).

### Risks / what could break

- **qb1/qb2 rsync target** is `~/tt-xla/` (HANDOFF.md §3-4 and every
  `serve*.sh`). The remote rename is a separate operation — recommend
  **keep the remote path as `~/tt-xla/`** initially (or rename remote too
  but only after local is verified). All sync scripts, server lifecycle
  docs, and Claude memory notes (`reference_tracy_build_qb1.md`,
  `feedback_*`) cite the remote path explicitly.
- Any in-flight `.cache/` files held open by editors.
- If Claude settings dir is renamed while a session is live, that session's
  JSONL becomes orphaned. Quit first.
- `pjrt_plugin/` build artifacts may have absolute rpaths embedded —
  rebuild required if you ever resume that arm.

---

## 3. GitHub repo rename

Current: `https://github.com/aweditya/tt-xla.git`.

### Candidate repo names

1. `tt-blackhole-llm` (matches recommended local name; readable on GitHub).
2. `qwen-blackhole-bringup` (descriptive of scope: Qwen + Blackhole +
   bring-up; longer but unambiguous).
3. `tt-metal-qwen36` (model-specific, signals tt-metal stack; rename again
   if other models become primary).

Recommendation: **`tt-blackhole-llm`** to keep local + remote in sync.

### Command (run from inside the repo)

```
gh repo rename tt-blackhole-llm
```

`gh repo rename` operates on the current repo via `gh`'s remote
detection. If invoked from inside the repo, it updates the GitHub side
**and** rewrites the `origin` URL in `.git/config` automatically; no
manual `git remote set-url` needed. GitHub serves a permanent HTTP 301
redirect from the old `aweditya/tt-xla` URL, so existing clones, PR
links, and `git fetch` from other machines continue to work transparently
(but updating them is hygienic).

If run from outside the repo, append the slug:
`gh repo rename tt-blackhole-llm -R aweditya/tt-xla`, then manually
`git remote set-url origin https://github.com/aweditya/tt-blackhole-llm.git`
inside each clone.

### Effects + verification

- `.git/config` `[remote "origin"]` URL flips to the new name (verify with
  `git remote -v`).
- GitHub Pages, Actions, webhooks: none configured in this repo; safe.
- Any external doc / Notion / Slack link pointing at the old repo follows
  the 301 redirect.
- The HANDOFF.md "GitHub repo" note (if present in any update) needs a
  one-line edit.

---

## Recommended order

1. Land the reorg PR first (preserves blame; one big move, all path
   refs updated in the same commit).
2. Then rename the GitHub repo via `gh repo rename` (zero local work).
3. Then quit Claude Code, rename the local folder + Claude settings dir,
   reopen.
4. Optionally rename the qb1/qb2 remote dirs in a separate session —
   higher risk because every script and every shipped memory note cites
   `~/tt-xla/`.
