# HANDOFF.md — TT-XLA Qwen3.6-27B Agent Onboarding

Last updated: 2026-05-18 evening (post owned_gdn default-flip). Authoritative entry point for any agent picking up this project. Read this end-to-end before touching code. Cite memory notes by filename when using older memory-bank findings, and cite in-repo research files for the current GDN work.

**Context-compacted agent?** Read `research/ACTIVE_CONTEXT.md` first. It is the
short active-state file for current measurements, invalidated paths, and next
actions. Do not spend tokens re-summarizing this whole handoff unless the user
explicitly asks for a comprehensive audit.

## Table of contents
0. [Current snapshot](#0-current-snapshot)
1. [What this project is](#1-what-this-project-is)
2. [Non-negotiables](#2-non-negotiables)
3. [Hosts (qb1 vs qb2)](#3-hosts)
4. [Persistent servers (lifecycle + endpoints)](#4-persistent-servers-lifecycle--endpoints)
5. [Repo layout](#5-repo-layout)
6. [Memory bank](#6-memory-bank)
7. [Roofline / ceiling math](#7-roofline--ceiling-math)
8. [Roadmap + shipped wins](#8-roadmap--shipped-wins)
9. [Reproducing key results](#9-reproducing-key-results)
10. [Tracing / profiling](#10-tracing--profiling)
11. [Web scraping / research strategy](#11-web-scraping--research-strategy)
12. [Background research strategy (4-agent parallel)](#12-background-research-strategy-4-agent-parallel-pattern)
13. [Plan-of-action template](#13-plan-of-action-template-7-step-workflow)
14. [Pitfalls (meta-lessons)](#14-pitfalls-meta-lessons)
15. [Friend repo (reference only)](#15-friend-repo-reference-only)
16. [Quick-start checklist](#16-quick-start-checklist-first-30-min-as-a-new-agent)
17. [Glossary](#17-glossary)

**Newcomer? Start with §0 + §16, then read §2, §3, §4, and §14 before writing any code.**

---

## 0. Current snapshot

If you are coming in cold or after context compaction, read this section and
`research/ACTIVE_CONTEXT.md` before touching any code.

### Active goal

The active goal is **Qwen3.6-27B decode performance optimization on
Tenstorrent Blackhole P150**, with the main focus on **multi-chip TP on qb2**.
The target is not a friend's number; it is to move measured end-to-end decode
as close to the hardware ceiling as correctness permits.

The current production path is:

- qb2, 4× P150, `(1,4)` mesh, `server_tp.py`, `MAX_POS=512`
- traced decode with paged SDPA, vocab-sharded LM head, on-device argmax,
  on-device embedding/cos/sin lookup, AND **the custom owned GDN kernel
  `ttnn.experimental.qwen36_gdn_decode_owned` as the default DeltaNet
  recurrence path** (commit `26cad39`, 2026-05-18 evening)
- measured end-to-end decode after the owned_gdn default flip: about
  **80 ms/token (~12.45 tok/s)** on `generate_tp` for the canonical prompts
  ("The capital of France is" → 80.26 ms/tok; "Implement a JSON parser
  combinator in Rust" → 80.44 ms/tok)

### Current bottleneck hypothesis

The 82 ms/token shape is not mostly Python/host overhead. The device trace is
large: 64 layers, many TTNN kernels, many layout/view operations, many
collectives, and a batch-1 decode graph with skinny GEMV-like matmuls. The
likely perf frontier is reducing small-op count and memory/layout traffic while
keeping correctness.

### Current custom GDN status — DEFAULTED 2026-05-18 evening

The custom single-device GDN kernel
`ttnn.experimental.qwen36_gdn_decode_owned` is now the production-default
DeltaNet recurrence path on qb2. Cold-bootstrap of `server_tp.py` captures
the owned kernel into the production trace; `generate_tp` runs at
**~80 ms/tok (~12.45 tok/s)** measured end-to-end.

Full diagnosis + gate-evidence trail:
`research/owned_gdn_diagnosis_2026_05_18.md`.

Headline mechanism: the owned kernel is **strictly more numerically accurate**
than the manual TTNN broadcast-reduce reference at the prediction step. The
manual path uses `ttnn.mul` (binary_ng) which, for all-bf16 inputs, runs
with `fp32_dest_acc_en=false` and so bf16-quantizes per-element products in
L1 before `ttnn.sum` reduces them in fp32 dst. The owned kernel keeps the
full contraction in fp32 dst across all 4 K-tile `matmul_tiles` calls and
packs once at the end. The two paths therefore differ by exactly 1 BF16
ULP at prediction (intrinsic, not a bug).

Files:
- `experiments/owned_ops/qwen36_gdn_decode_owned/` (the fused op + tests +
  benchmark + microbench harness)
- `experiments/owned_ops/qwen36_gdn_{decay_state,prediction,delta,outer_update,output}/`
  (the five component ops the fused op was built from)
- `experiments/serve/server_tp.py:115-121` (`deltanet_recurrence_mode`
  default = `"owned_gdn"`)

Gates passed:
- ULP-aware tensor at layer 0 (prediction max_diff ≤ 1 BF16 ULP, state ≤ 2 ULP).
- Teacher-forced argmax on 3 prompts / 236 positions: **230/236** (97.5%);
  all 6 flips are sub-quarter-logit razor ties the manual reference also has
  at adjacent positions.
- Tier 1 long-context (200 positions): **0/200** disagreements.
- Tier 3 long-context (500 positions, matches qb1 P21 bar): **10/500 = 2.0%**
  disagreements, median cosine 0.9995, NO cliff (rolling 50-step medians flat
  at 0.999 across all 500 steps).
- Measured production decode: 80.26-80.44 ms/tok vs 82.92 ms/tok manual
  baseline = **+3.2% perf**.

Key artifacts (correctness):
```text
.cache/qb2_tp_deltanet/cosine_ladder_tp_compare_20260518_2105.json   (Tier 1: 200 pos, 0/200)
.cache/qb2_tp_deltanet/cosine_ladder_tp_compare_500_20260518.json    (Tier 3: 500 pos, 10/500)
research/owned_gdn_diagnosis_2026_05_18.md                            (full diagnosis + gate trail)
```

Known follow-up tracked, not promotion-blocking:
- Eager `owned_gdn` slowdown on the 2nd+ invocation per server lifetime
  (commit `2905470`). Production decode is traced and unaffected; eager
  probes that toggle modes need server restart between owned_gdn runs.

Rollback: set `state.deltanet_recurrence_mode = "manual"` in
`MeshServerState.__init__` (`server_tp.py:115`) and re-bootstrap.

### Current remote state

As of the last update in this handoff, qb2 resident server was restored and
healthy:

```bash
ssh qb2 'cd ~/tt-xla && .venv/bin/python -m experiments.serve.client_tp status'
```

Do not assume that remains true. Check status first.

### Non-current / parked

- PJRT plugin is parked for the distant future. Keep it clean, but do not
  prioritize it over Qwen3.6-27B bring-up/perf.
- Friend repo is reference-only. It contains useful TT-Metal patterns and also
  known/possible GDN errors. Do not treat it as ground truth.

## 1. What this project is

Stanford **CS440LX exploratory research**: build deep understanding of Tenstorrent Blackhole P150 hardware + JAX/XLA internals, then eventually ship a JAX/XLA backend for Tenstorrent. Active line of work is bringing up and optimizing **Qwen3.6-27B** (hybrid recurrent: DeltaNet/GDN + Gated Attention + dense MLP, 64 layers, ~27B params) on **P150**.

The 27B bring-up is the proving ground. Once correctness and perf are saturated, learnings can flow back into the PJRT plugin (`pjrt_plugin/`). For now, prioritize model bring-up and performance optimization over PJRT work.

---

## 2. Non-negotiables

Hard rules. Violating them silently wastes hours.

| # | Rule | Rationale |
|---|---|---|
| 1 | **Correctness before performance** | Do not optimize, integrate, or default-enable a path that has not passed its equivalence gate. |
| 2 | **Hypothesis-driven workflow** | Every claim grounded in experiment. Record rejected hypotheses and artifacts. |
| 3 | **No code bloat** | Think more, type less. Concise, scoped changes only. |
| 4 | **Remote execution only** — `ssh qb1` / `ssh qb2` | The local Mac has no Tenstorrent hardware. `ssh tenstorrent` (legacy) is GONE. |
| 5 | **qb2 is the multi-chip target** | Main agent work is multi-chip TP/perf on qb2. Use qb1 for single-chip only when available. |
| 6 | **No inline scripts** (`python -c '...'`) | Always write a permanent helper in `experiments/utils/` and run that. See `reference_inline_script_helpers.md`. |
| 7 | **No `/tmp`** | All scratch goes in project dirs — `.cache/`, `experiments/`, `research/`, `wiki/`. |
| 8 | **Frequent commits** | Auto-commit allowed (`feedback_commits.md`). Watchdog kills sessions silent > 600s. |
| 9 | **No local execution of device code** | Local Mac is read/write/edit + git only. No `ttnn` imports locally. |
| 10 | **No hallucinated improvements** | Do not claim `X ms` or `%` speedup unless measured end-to-end or theoretically justified and clearly labeled as a bound. |
| 11 | **Numpy oracle, not HuggingFace AutoModel** | `AutoModel.from_pretrained` crashes on remote; build pure-numpy fp32 reference + construct `DecoderLayer` directly with `safe_open` weights (`reference_hf_oracle_pattern.md`). |
| 12 | **Sync-bounded timing** | ttnn dispatch is async. `ttnn.synchronize_device(device)` BEFORE start AND AFTER stop, every benchmark (`feedback_sync_bounded_timing.md`). |
| 13 | **Stdout line-buffering for SSH probes** | SSH pipes are block-buffered. Add `sys.stdout.reconfigure(line_buffering=True)` at top of every helper (`feedback_python_stdout_buffering.md`). |
| 14 | **Doc-first** | Before any non-trivial ttnn/mesh/fabric work, read `experiments/.refs/tt-metal/tech_reports/` + `models/demos/llama3_70b_galaxy/`. Cite doc path in design (`feedback_consult_docs_before_acting.md`). |
| 15 | **Never cite projection as measurement** | Per-block × layer count is a CEILING not tok/s. Real numbers require full `bench_decode` (`feedback_real_vs_projected.md`). |
| 16 | **Resident server owns the chips** | If a server is up, use its socket endpoints. Do not start raw TTNN probes on that host unless the user approves stopping/restarting the server. |
| 17 | **Custom GDN gate** | No full GDN integration until isolated contraction, full recurrence, teacher-forced tokens, generated tokens, and perf are all validated. |

---

## 3. Hosts

Two hosts, both active concurrently. Each has 4× Tenstorrent Blackhole P150. **There are NO NVIDIA GPUs.** Check with `tt-smi -s` (`-s` snapshot flag is required; tt-smi defaults to interactive TUI which hangs SSH).

| Host | Chips | Fabric | Use case | Bootstrap time |
|------|-------|--------|----------|----------------|
| `qb1` | 4× P150 | NO inter-chip fabric | **Single-chip only.** Server `server.py`. Long-context drift work, single-chip GDN/perf probes, Tracy profiling. Ask user before using if they are running experiments. | ~11 min weight load |
| `qb2` | 4× P150 | Working fabric | **Primary target.** Multi-chip TP, profiling, all GDN integration checks that touch TP. Server `server_tp.py`. `(1,4)` mesh, all_gather/all_reduce, distributed layers. | ~5-10 min sharded load |

```
❌ WRONG                              ✓ RIGHT
nvidia-smi                            ssh qb1 'tt-smi -s'
CUDA_VISIBLE_DEVICES=0                device_ids=[0] in ttnn (but see caveat)
torch.cuda.empty_cache()              ttnn.synchronize_device(device)
reboot                                ssh qbX 'tt-smi -r 0,1,2,3'
```

**Critical caveat (`feedback_mtp_head_probe.md`):** ttnn opens ALL 4 chips even with `device_ids=[N]`. You **cannot run two ttnn processes concurrently on the same host**. If the persistent server is up, no other ttnn probe can run on that host — you'll get SIGBUS or hang. Ask the user before stopping a server.

**Mesh recovery (qb2 only):** if a mesh process gets `pkill -9`'d, fabric state corrupts. Recover with `ssh qb2 'tt-smi -r 0,1,2,3'` (`feedback_mesh_recovery_after_kill.md`).

---

## 4. Persistent servers (lifecycle + endpoints)

Both servers use a **Unix domain socket**, NOT HTTP. Use `experiments/serve/client.py` (or `client_tp.py` on qb2), NOT `curl`.

### Sockets

| Host | Server | Socket path | Protocol |
|------|--------|-------------|----------|
| qb1 | `experiments/serve/server.py` | `~/tt-xla/.cache/server.sock` | `experiments/serve/protocol.py` (JSON lines) |
| qb2 | `experiments/serve/server_tp.py` | `~/tt-xla/.cache/server_tp.sock` | same protocol |

Wire format: newline-terminated JSON. `pack_request(cmd, args)` + `read_line` + `parse_response`. Streaming responses (`generate_long`, `generate_stream`, `cosine_ladder`) emit multiple `chunk` frames then a final `result` frame.

### Lifecycle

| Action | qb1 | qb2 |
|--------|-----|-----|
| Start | `ssh qb1 'bash ~/tt-xla/experiments/serve/scripts/serve.sh start'` | `ssh qb2 'bash ~/tt-xla/experiments/serve/scripts/serve_tp.sh start'` |
| Stop  | `... serve.sh stop` | `... serve_tp.sh stop` |
| Status | `... serve.sh status` | `... serve_tp.sh status` |
| Tail log | `ssh qb1 'tail -f ~/tt-xla/.cache/server.log'` | `ssh qb2 'tail -f ~/tt-xla/.cache/server_tp.log'` |

### Endpoint surface (qb1 — confirmed via `grep handle_ experiments/serve/server.py`)

| Cmd | Purpose |
|-----|---------|
| `status` | Server health |
| `reset_state` | Clear KV cache |
| `reload_kernels` | Invalidate trace caches |
| `run_91r` | Per-layer cosine sanity (vs numpy oracle) |
| `bench_decode` / `bench_decode_paged` / `bench_decode_traced` | Perf benchmarks |
| `cosine_ladder` | Per-position teacher-forced logits vs fp32 oracle (long-context drift) |
| `generate` | Short prompt completion (greedy default) |
| `generate_paged` / `generate_long` | Long-context streaming completion |
| `generate_stream` | Token-by-token streaming |
| `shutdown` | Clean teardown |

qb2 (`server_tp.py`) is now the main multi-chip development surface. It has
`generate_tp`, decode profiling/benchmark endpoints, RoPE/collective probes,
and multiple DeltaNet/GDN probes. Verify the current surface with:

```bash
rg -n '^def handle_|HANDLERS' experiments/serve/server_tp.py
ssh qb2 'cd ~/tt-xla && .venv/bin/python -m experiments.serve.client_tp status'
```

Prefer resident-server probes on qb2. Raw TTNN scripts contend with the server
because TTNN opens all chips.

### Sample client call (from a probe helper)

```python
from experiments.serve import protocol as P
import socket
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect(P.SOCKET_PATH)  # qb1: server.sock — for qb2 swap to ~/tt-xla/.cache/server_tp.sock
sock.sendall(P.pack_request("generate", {"prompt": "The capital of France is", "max_tokens": 20}))
resp = P.parse_response(P.read_line(sock, max_bytes=64 << 20))
print(resp.data)
```

---

## 5. Repo layout

```
tt-xla/
├── CLAUDE.md                 # project rules (READ FIRST after this file)
├── HANDOFF.md                # this file
├── PLAN.md / REPRODUCE.md    # legacy plan + reproduction docs
├── pjrt_plugin/              # the actual JAX backend (PJRT plugin in C++/Python)
│   ├── scripts/
│   ├── tests/
│   └── ...
├── wiki/                     # Q&A wiki — learning-by-building
├── research/                 # raw notes, scraped content, integration outlines
│   └── ACTIVE_CONTEXT.md      # current active-state handoff for compactions
├── experiments/              # all device code (runs on qb1/qb2)
│   ├── 01_jax_basics.py ... 99_moe_dram_sharded.py   # numbered experiments (chronological)
│   ├── 91*_qwen36_27b_*.py                            # 27B bringup series (CURRENT FOCUS)
│   ├── demo_qwen36_27b.py
│   ├── serve/                # persistent inference servers
│   │   ├── server.py         # qb1 single-chip
│   │   ├── server_tp.py      # qb2 multi-chip TP
│   │   ├── protocol.py       # Unix socket JSON-line wire protocol
│   │   ├── client.py / client_tp.py
│   │   ├── scripts/          # serve.sh, serve_tp.sh, drift sweeps
│   │   └── tests/
│   ├── utils/                # permanent probe helpers (no inline scripts!)
│   ├── owned_ops/            # custom TTNN/TT-Metal op bring-up; current GDN work lives here
│   ├── tt_jax/               # tt-metal -> JAX integration scaffolding
│   ├── .refs/                # vendored tt-metal references (read-only)
│   └── logs/
├── demos/                    # presentation demos
└── tt_docs_corpus/           # scraped tt-metal / tenstorrent docs
```

Current focus: `experiments/serve/server_tp.py`,
`experiments/owned_ops/qwen36_gdn_decode_owned/`,
`experiments/owned_ops/qwen36_gdn_prediction/`, and
`research/ACTIVE_CONTEXT.md`. The single most-touched production file is
**`experiments/serve/server_tp.py`** (qb2 multi-chip).

---

## 6. Memory bank

Auto-memory lives at:
```
~/.claude/projects/-Users-adityasriram-Labs-stanford-cs440lx-tt-xla/memory/
```

The historical index is `MEMORY.md`. It had ~140 entries as of 2026-05-14 and
has since been supplemented by in-repo current-state files such as
`research/ACTIVE_CONTEXT.md` and the `experiments/owned_ops/*/README.md`
documents. Use the in-repo docs first for GDN/current TP state.

| Group | Examples | Topic |
|-------|----------|-------|
| **Branch III** | `project_branchIII_27b_complete.md`, `feedback_isolation_must_match_production.md` | 27B bringup, correctness ladder |
| **Branch C'** | `project_branchC_perf_state.md`, `feedback_paged_sdpa_decode_works_at_32k.md` | Single-chip perf |
| **Branch C'7** | `feedback_c71_mesh_smoke_pass.md` ... `feedback_c77_real_weights_tp_pass.md` | Multi-chip TP build-up |
| **Branch D'** | `feedback_speculative_decoding.md`, `feedback_mtp_head_probe.md` | MTP / spec decode probes |
| **Long context** | `feedback_bf16_prefill_drift_cliff.md`, `feedback_fp32_sdpa_cliff_probe.md` | Drift root-cause + fix |
| **Multi-chip shipped** | `feedback_paged_sdpa_shipped_tp.md` | 11.43 tok/s win |
| **Reference** | `reference_how_to_run_stuff.md`, `reference_tracy_build_qb1.md`, `reference_hf_oracle_pattern.md` | Standing rules + patterns |

**Cite by filename.** When a finding from a memory note is relevant, mention `feedback_X.md` in your reasoning. Future agents will read the trail.

**Read `MEMORY.md` end-to-end** at session start. The index is the single most-valuable artifact in this project — it captures every dead-end, every gotcha, every micro-optimization measurement so you don't re-run them.

---

## 7. Roofline / ceiling math

From `reference_p150_roofline_priority.md` and `feedback_realistic_tp_ceiling.md`:

| Quantity | Value | Source |
|---|---|---|
| P150 peak DRAM bandwidth | 512 GB/s | Tenstorrent docs |
| Qwen3.6-27B weight footprint at bf16/bf8 mix | ~27 GB shipping; ~21 GB resident per chip on TP4 | empirical |
| Single-chip baseline (eager, 91f) | 200.81 ms/tok = 4.98 tok/s | `feedback_qk_rms_norm_shipped.md` precursor |
| Single-chip post QK rms_norm fusion | 192.81 ms/tok = 5.19 tok/s | `feedback_qk_rms_norm_shipped.md` |
| TP traced baseline (pre-paged-SDPA) | 7.02 tok/s | commit `9369e1b` |
| **TP shipped (paged SDPA + HiFi2 B3)** | **11.43 tok/s** | `feedback_paged_sdpa_shipped_tp.md`, commit `4741253` |
| Friend's daily-driver | ~15.3-15.5 tok/s | reference point only, not the target |
| El Reg Llama-3.1-70B 4×P150 measured | 1.78× TP4 vs TP1 (~41% of theoretical) | `feedback_realistic_tp_ceiling.md` — realistic ceiling, not 4× |

### Targets (single-chip, bf16/bf8 mix, MAX_POS=256)

| ms/tok | tok/s | % of 512 GB/s peak |
|---|---|---|
| 350 | 2.86 | 15% |
| 250 | 4.0 | 21% |
| 200 (current single-chip) | 5.0 | 27% |
| 150 | 6.67 | 36% |
| 100 | 10.0 | 54% |
| ~55 (peak) | 18.7 | 100% |

### Where the gap is (multi-chip, `feedback_tracy_tp_breakdown.md`)

Of the 142 ms/tok at 7.02 tok/s pre-paged-SDPA:
- 77 ms compute (1.21 ms/block × 64 — scales as expected)
- 54 ms gap: 16× gated_attn blocks (manual SDPA on mesh + extra collectives + RoPE)  ← fixed in paged-SDPA ship
- 9.4 ms logits readback
- 1.9 ms update_input_buffers
- ~0 ms sync barrier

The remaining lever set is the **multi-chip TP opt menu** (`research/multi_chip_optimizations_menu.md` + `_v2_addendum.md`) plus the post-P22/P25 memory notes. Vocab-sharded lm_head and on-device embed/cos/sin lookup have shipped. Distributed RMSNorm and DRAM-sharded MLP were probed negative at current shapes. Current likely levers are fresh profiling, CCL cleanup/overlap, native RoPE, and deeper DeltaNet/GDN fusion, but every candidate must be validated against full decode and correctness gates.

---

## 8. Roadmap + shipped wins

### Branch structure (chronological)

| Branch | Theme | Key memory notes |
|--------|-------|------------------|
| **Branch III** | 27B bringup, correctness ladder, 7 bugs fixed | `project_branchIII_27b_complete.md`, `feedback_isolation_must_match_production.md` |
| **Branch C'** | Single-chip perf — fusion, traces, paged KV cache | `project_branchC_perf_state.md`, `feedback_c4v4_validated.md`, `feedback_qk_rms_norm_shipped.md` |
| **Branch C'7** | Multi-chip TP — mesh, sharding, all_reduce, trace on mesh | `feedback_c71_mesh_smoke_pass.md` ... `feedback_c761_tp_trace_wins_big.md` |
| **Branch D'** | Speculative decoding probes (MTP head, B=2 verify) | `feedback_speculative_decoding.md`, `feedback_mtp_head_probe.md`, `feedback_d3_dont_ship_yet.md` |
| **Long context** | bf16 prefill drift root-cause + fix | `feedback_bf16_prefill_drift_cliff.md`, `feedback_fp32_sdpa_cliff_probe.md` |

### Shipped wins (latest first — `git log --oneline -25`)

| Commit | Title | Impact |
|--------|-------|--------|
| `56de5d2` | P25 on-device embed + cos/sin lookup | **12.02 → 12.07 tok/s (+0.4%)** |
| `e71f9b2` | P25 probe: on-device embed + plus_one + cos/sin lookup | probe |
| `24e07a8` | P24 distributed RMSNorm probe | negative: correct but slower |
| `8a55933` | P22 post-ship bench data | **12.02 tok/s** |
| `ef3f336` | SHIP P22 vocab-sharded lm_head + on-device argmax | **11.43 → 12.02 tok/s (+5.1%)** |
| `dada0a5` | P23 DRAM-sharded MLP probe | negative: 2.1× slower |
| `4741253` | SHIP V2 paged SDPA on (1,4) mesh | **7.02 → 11.43 tok/s (62% gain)** |
| `57b1a4e` | Friend repo borrow list + web research v2 | research |
| `8abb089` | P21 cliff probe: HiFi4+fp32_dest_acc is the bug; HiFi2 unblocks long-context | drift fix |
| `94f7cba` | P19: paged SDPA chained-context (eager +10.2%, traced wash) | exploratory |
| `98189be` | bf16 prefill drift cliff: pos 129 cos<0.9, pos 141 cos<0.5 | drift root-cause |
| `09c935b` | Distributed RMSNorm integration outline | design |
| `99aaa87` | Integration outline: vocab-sharded LM head | design |
| `0593245` | P18 mesh paged SDPA: WORKS at production shapes — 3.59× | unblock |
| `844e194` | Needle-in-haystack at L=500+ — qb1 prefill fully broken | failure mode |
| `9e75310` | TP per-step decomposition: compute IS scaling 4× | analysis |
| `9369e1b` | TRACED multi-chip TP inference — 7.02 tok/s correct output | first TP ship |
| `2d30af7` | P14: TRACE CAPTURE WORKS ON (1,4) MESH | unblock |
| `4cd0ce1` | server_tp.py: paged KV cache refactor — enables trace | enabler |

Latest shipped wins (today, 2026-05-18 evening):

| Commit | Title | Impact |
|--------|-------|--------|
| `26cad39` | **DEFAULT owned_gdn**: production trace now captures the owned GDN kernel | **12.06 → 12.46 tok/s (+3.2%)** measured |
| `040e2ac` | Tier 3 long-context gate PASSES (500 positions, no cliff) | promotion-gate |
| `b4c62ab` | qb2 MAX_POS 256 → 512 (matches qb1 single-chip 500-position bar) | enabler |
| `e9b4b06` | Tier 1 long-context gate PASSES (200 positions, 0/200) | promotion-gate |
| `088c33b` | qb2 `cosine_ladder_tp` endpoint + client wrapper | enabler |
| `27a91e4` | Owned GDN diagnosis memo (kernel > reference accuracy) | mechanism understood |
| `d3423f6` | Owned GDN op trees (5 components + fused decode_owned) | first owned TT-Metal op |

### Current state (as of 2026-05-18 evening, post owned_gdn default)

| Metric | Value | Note |
|--------|-------|------|
| Production perf (qb2 multi-chip TP traced + owned_gdn) | **~12.45 tok/s / ~80 ms-tok** | measured 2026-05-18 evening on 2 prompts × 30 tok; commit `26cad39` |
| Production perf (qb1 single-chip) | 5.19 tok/s | `feedback_qk_rms_norm_shipped.md` |
| Long-context cliff (qb1, HiFi2 B3) | none up to L=500 | `feedback_fp32_sdpa_cliff_probe.md` |
| Long-context cliff (qb2, owned_gdn vs manual) | none up to L=500 (10/500 razor-tie flips, no cliff) | `cosine_ladder_tp_compare_500_20260518.json` |
| `MAX_POS` ceiling (qb2 trace) | **512** (was 256 pre-2026-05-18 evening) | `b4c62ab` |
| Speculative decode (D'3) | DON'T SHIP at 57.9% acceptance | `feedback_d3_dont_ship_yet.md` |
| El Reg ceiling (4× P150 Llama70B BFP8) | 1.78× (~9.2 tok/s equivalent) | `feedback_realistic_tp_ceiling.md` — we're above this |
| Optimization target | approach hardware ceiling | compare against roofline / measured full-decode tok/s |
| Custom GDN status | **DEFAULT** (commit `26cad39`) | see §0 and `research/owned_gdn_diagnosis_2026_05_18.md` |

### Next investigation list (post owned_gdn default, 2026-05-18 evening)

1. **Fresh full-decode/profile breakdown with owned_gdn defaulted.** The pre-owned profiles (`results_decode_op_counts_20260515_0129.json`) attributed DeltaNet recurrence as ~17% of eager-op time. The owned kernel collapsed 480 recurrence-category ops into 1; the new bottleneck distribution must be re-measured before picking the next fusion target.
2. **Apply the kernel-fusion pattern to the next op-heavy region.** Top remaining candidates from the pre-owned-gdn profile: DeltaNet decay/gate (480 calls/token), DeltaNet QKV repeat (336), RoPE chain (320), attention plumbing (320). Each is a multi-week custom-op project on the same scaffolding as `qwen36_gdn_decode_owned`.
3. **Tier 4 daily-driver-length extension.** Bump `MAX_POS` to 1024 or 2048 and re-run `cosine_ladder_tp` at the longer length to confirm no cliff emerges past 500. Same pattern as Tier 3 (commit `040e2ac`); ~30 min wall on qb2.
4. **Root-cause the eager owned_gdn slowdown** (`2905470`). Production-irrelevant but blocks future eager probe development.
5. **Native RoPE / fewer RoPE dispatches.** Manual rotate-only remains; the slice-first `ttnn.experimental.rotary_embedding` recipe passed an isolated correctness gate (`results_native_partial_pass_20260515_0030.json`) but was never landed in production.
6. **CCL cleanup / overlap.** No evidence yet that collectives overlap with compute. Tracy on qb2 (build at `~/tenstorrent/tt-metal/build_tracy_gcc12_nodist`) is set up; needs a real per-op pass.
7. **Do not re-open killed candidates without new evidence:** distributed RMSNorm (`feedback_distributed_rms_norm_failed.md`), single-P150 DRAM-sharded MLP (`feedback_dram_sharded_mlp_probe.md`), strict-reduce owned-GDN path (rejected 2026-05-17, see `cache/qb2_tp_deltanet/qwen36_gdn_decode_owned_strict_*.json`).

---

## 9. Reproducing key results

### Cold-start the qb2 multi-chip server (~5-10 min bootstrap)

```bash
$ ssh qb2 'bash ~/tt-xla/experiments/serve/scripts/serve_tp.sh start'
$ ssh qb2 'tail -f ~/tt-xla/.cache/server_tp.log'   # wait for "Stage A/B/C/D/E ready"
$ ssh qb2 'bash ~/tt-xla/experiments/serve/scripts/serve_tp.sh status'
```

### Reproduce current qb2 TP smoke/perf (P25 line)

Server must be up on qb2. Then:

```bash
$ ssh qb2 'cd ~/tt-xla && .venv/bin/python -m experiments.serve.client_tp \
    generate_tp --prompt "The capital of France is" --max-tokens 30 --chunk-size 30'
# expected: coherent "Paris" continuation and ~82.8 ms/tok decode (~12.07 tok/s)
```

Quick smoke (correctness):

```bash
$ ssh qb2 'cd ~/tt-xla && .venv/bin/python -m experiments.serve.client_tp \
    generate_tp --prompt "The capital of France is" --max-tokens 10'
# expected: " Paris." continuation
```

### Reproduce current GDN equivalence state (qb2 resident server)

Server must be up on qb2. These run through the resident server, not raw TTNN.
Do not run `--component-debug-mode 2` unless you are prepared to restart the
server; it has tied up the resident path before.

```bash
# Main real-tensor stepwise probe with the clean matmul-contract check.
$ ssh qb2 'cd ~/tt-xla && .venv/bin/python -m experiments.serve.client_tp \
    probe_deltanet_owned_gdn_real_tensors_tp \
    --prompt "The capital of France is" \
    --layer-idx 0 --native-io --stepwise --seed-state manual_once \
    --output-json .cache/qb2_tp_deltanet/results_owned_gdn_matmul_contract_nativeio_seeded_l0_20260518.json'

# K materialization sanity: should be exact.
$ ssh qb2 'cd ~/tt-xla && .venv/bin/python -m experiments.serve.client_tp \
    probe_deltanet_owned_gdn_real_tensors_tp \
    --prompt "The capital of France is" \
    --layer-idx 0 --native-io --stepwise --seed-state manual_once \
    --component-debug-mode 10 \
    --output-json .cache/qb2_tp_deltanet/results_owned_gdn_component_mode10_kcol_nativeio_seeded_l0_20260518.json'

# Product/reduce diagnostics: close but not default-safe.
$ ssh qb2 'cd ~/tt-xla && .venv/bin/python -m experiments.serve.client_tp \
    probe_deltanet_owned_gdn_real_tensors_tp \
    --prompt "The capital of France is" \
    --layer-idx 0 --native-io --stepwise --seed-state manual_once \
    --component-debug-mode 11 \
    --output-json .cache/qb2_tp_deltanet/results_owned_gdn_component_mode11_product_ttnn_expected_nativeio_seeded_l0_20260518.json'

$ ssh qb2 'cd ~/tt-xla && .venv/bin/python -m experiments.serve.client_tp \
    probe_deltanet_owned_gdn_real_tensors_tp \
    --prompt "The capital of France is" \
    --layer-idx 0 --native-io --stepwise --seed-state manual_once \
    --component-debug-mode 12 \
    --output-json .cache/qb2_tp_deltanet/results_owned_gdn_component_mode12_reduce_ttnn_expected_nativeio_seeded_l0_20260518.json'
```

Expected current conclusions:

- `component_kcol0` exact.
- `component_prediction` max diff `0.0078125` vs broadcast-reduce reference.
- `prediction_matmul_vs_broadcast` max diff `0.0625`; matmul is shape-valid
  but numerically different from the existing recurrence contract.
- Full owned GDN `pass_gate: false`.

### Reproduce single-chip 5.19 tok/s (qb1)

```bash
$ ssh qb1 'bash ~/tt-xla/experiments/serve/scripts/serve.sh start'   # ~11 min
$ ssh qb1 'cd ~/tt-xla && .venv/bin/python -m experiments.serve.client \
    bench_decode_traced --tokens 32 --runs 3'
```

### Reproduce long-context fix (HiFi2 B3 vs prod HiFi4)

Sentinel-driven, no re-bootstrap needed:

```bash
$ ssh qb1 'echo B3 > ~/tt-xla/.cache/p21_sdpa_variant.txt'
$ ssh qb1 'cd ~/tt-xla && .venv/bin/python -m experiments.serve.client \
    reload_kernels'
$ ssh qb1 'cd ~/tt-xla && .venv/bin/python experiments/utils/needle_haystack_probe.py'
# expected: variant B3 retrieves D827W4MW at L=500 frac=0.5
```

### Reproduce cosine ladder (drift root-cause, 500 positions)

```bash
$ ssh qb1 'cd ~/tt-xla && .venv/bin/python -m experiments.serve.client \
    cosine_ladder --max-pos 500 --prompt-file prompts/drift.txt'
# expected (HiFi4 prod variant A): cliff at pos 129
# expected (HiFi2 B3 variant): no cliff up to 500
```

### Verify a probe before running it

```bash
$ ssh qb1 'cd ~/tt-xla && .venv/bin/python experiments/utils/syntax_check.py \
    experiments/utils/<your_probe>.py'
# also: ssh qb1 'tt-smi -s' to verify chips are healthy first
```

---

## 10. Tracing / profiling

### Tracy / profiling target split

For **multi-chip TP profiling**, qb2 is the target. Start with resident-server
profiling/breakdown helpers so you do not contend with the server:
`experiments/utils/tp_decompose_probe.py`, the profiling endpoints in
`experiments/serve/server_tp.py`, and any current qb2-specific helper in
`experiments/utils/`.

Known Tracy setup:

- Tracy-enabled build: `~/tt-metal-tracy/` on qb1
- Wrapper script: `run_with_tracy_build.sh`
- Use this for single-chip Tracy comparisons unless/until qb2 has a verified
  Tracy-capable build.
- The default ttnn build's Tracy API is a no-op. Setting
  `TT_METAL_DEVICE_PROFILER=1` on the non-Tracy build aborts
  (`feedback_ttnn_device_profiler_build.md`).
- Historically qb2 did not have the same Tracy setup as qb1. Verify the current
  qb2 profiling toolchain before assuming device-side Tracy is available.

```bash
$ ssh qb1 'bash ~/tt-xla/run_with_tracy_build.sh experiments/utils/<probe>.py'
```

### Current profiling conclusion

The current multi-chip decode shape should be treated as a **device graph**
problem, not a Python-loop problem. The working hypothesis from recent P25
profiling/discussion is:

- batch-1 decode is skinny, so many matmuls are GEMV-like and underutilize the
  machine
- trace replay removes Python overhead but does not fuse many TTNN kernels
- DeltaNet recurrence, RoPE/QK plumbing, layout/view ops, cache updates,
  collectives, and LM-head path all contribute
- full-attention layers are only 16/64, so fixing SDPA/KV alone cannot explain
  the full remaining gap

The next profile should produce a current per-op breakdown grouped into:
matmul, RMSNorm, RoPE, DeltaNet recurrence, collectives, cache update, SDPA,
and LM head. Use that breakdown to choose fusion targets; do not guess.

### Sync-bounded timing (`feedback_sync_bounded_timing.md`)

```python
import time, ttnn
ttnn.synchronize_device(device)          # critical pre-sync
t0 = time.perf_counter()
out = my_op(...)
ttnn.synchronize_device(device)          # critical post-sync
elapsed_ms = (time.perf_counter() - t0) * 1e3
```

Without both syncs, you measure DISPATCH latency, not EXECUTE latency. Async launches make this trap easy to fall into.

### `bench_decode` family

`bench_decode` (eager), `bench_decode_paged` (eager paged), `bench_decode_traced` (trace replay) all wrap the full decode loop including I/O. Never report `execute_trace`-only timing as tok/s (see `feedback_benchmark_methodology.md`, `feedback_real_vs_projected.md`).

### Tracy analysis helpers (in tree)

- `experiments/utils/tracy_traced_decode_probe.py` — capture a Tracy run
- `experiments/utils/tracy_analyze_ops.py` — parse Tracy output → per-op time
- `experiments/utils/tracy_top_ops_breakdown.py` — top-N table
- `experiments/utils/tp_decompose_probe.py` — per-component TP step breakdown (qb2)

---

## 11. Web scraping / research strategy

The local research corpus has already been built. **Don't re-scrape unless the question is genuinely novel.**

### Existing research index

| Topic | Path |
|-------|------|
| tt-metal source tree (vendored) | `experiments/.refs/tt-metal/` — read directly, don't re-fetch |
| tt-metal tech reports | `experiments/.refs/tt-metal/tech_reports/` — START HERE for ttnn ops |
| Scraped Tenstorrent docs corpus | `tt_docs_corpus/` (versioned) |
| Multi-chip web research v1 | `research/multi_chip_web_research.md` (Agent P) |
| Multi-chip web research v2 | `research/multi_chip_web_research_v2.md` (Agent Y) |
| Friend's daily-driver repo, scraped | `research/friend_repo_borrow_list.md` — REFERENCE ONLY (see §15) |
| Multi-chip opt menu v1 | `research/multi_chip_optimizations_menu.md` (14 ranked candidates) |
| Multi-chip opt menu v2 | `research/multi_chip_optimizations_menu_v2_addendum.md` (11 more candidates) |
| PJRT reflections | `research/pjrt_reflections.md` |
| Branch reflections | `research/branch_*.md` |

### Key external URLs (already mined — see `reference_research_sources.md`)

- https://tenstorrent.com (corporate, product info)
- https://www.corsix.org/content/tt-wh-part1 (Corsix 8-part Wormhole series)
- https://clehaxze.tw/gemlog/2025/04-21-programming-tensotrrent-processors.gmi
- https://github.com/tenstorrent/tt-metal (canonical kernels)
- https://github.com/jax-ml/jax + https://github.com/openxla/xla
- El Reg 4× P150 QuietBox review (Nov 2025): https://www.theregister.com/2025/11/27/tenstorrent_quietbox_review/
- ASPLOS 2025 "Dissecting Blackhole": https://asplos.dev/wordpress/wp-content/uploads/2025/09/TT_bench-1.pdf
- tt-metal #26252 (all_gather BW analysis), #33147 (CCL scaling tuning knobs)

### Rule

Before scraping anything new, **grep the existing research/ + tt_docs_corpus/ + .refs/ first**. Most questions are answered locally. `feedback_consult_docs_before_acting.md` is a hard rule: cite a doc path before designing a fix.

---

## 12. Background research strategy (4-agent parallel pattern)

The project uses a **fan-out research pattern** where multiple sub-agents work concurrently on a question, then converge findings into a memory note. Common in the Branch C'7 era.

### Pattern

1. **User asks a big question** (e.g. "what TP optimizations exist?")
2. **Spawn 2-4 agents** with non-overlapping mandates (e.g. Agent O = local docs survey, Agent O2 = deeper docs dive, Agent P = web research, Agent Y = competitor benchmarks)
3. **Each agent writes a separate research note** (`research/multi_chip_optimizations_menu.md`, `_v2_addendum.md`, `multi_chip_web_research.md`, `multi_chip_web_research_v2.md`)
4. **Converge into a single memory note** (e.g. `reference_multi_chip_opt_menu.md`, `reference_multi_chip_web_research.md`)

### Naming convention

- Agent letters track contribution: O, O2, P, Q, V, W, X, Y, K, N — referenced inline in memory notes ("per Agent Y's finding")
- File suffix `_v2_addendum.md` or `_v2.md` means another agent extended an earlier doc

### Contention rules

- **Never run two ttnn probes on the same host concurrently** (see §3). If one agent is on qb1, another must use qb2 or local-only research.
- Persistent server claims its host's ttnn — agents needing raw device access must coordinate.
- Memory writes are safe — multiple agents can append to `research/` simultaneously; the MEMORY.md index gets merged at session end.

### When to fan-out vs sequential

| Fan-out when | Sequential when |
|--------------|------------------|
| Question has 2+ independent sub-questions | Each step depends on the prior |
| Mixed local-only + remote work possible | All work needs the same scarce resource (one host) |
| 90-min+ research arc | <30 min task |

---

## 13. Plan-of-action template (7-step workflow)

Follow this for any non-trivial task. Cut corners only when the user explicitly says so.

| # | Step | Output | Time |
|---|------|--------|------|
| 1 | **Read MEMORY.md index** | Identify relevant `feedback_*.md` notes | 5 min |
| 2 | **Read those notes** (3-5 typically) | Know prior findings, gotchas, dead-ends | 10 min |
| 3 | **Check `experiments/.refs/tt-metal/`** for canonical implementation | Cite doc path before designing | 10 min |
| 4 | **Write a plan** (design memo or `research/<topic>.md`) | Hypothesis, experiment design, success criteria | 15 min |
| 5 | **Write a probe helper** in `experiments/utils/<name>.py` (NEVER inline `python -c`) | Permanent, line-buffered, sync-bounded | 30 min |
| 6 | **Run probe**, capture results in `.cache/<probe_name>/`, **commit** | Data + log | 15-90 min |
| 7 | **Write memory note** `feedback_<finding>.md` + update `MEMORY.md` index | Future agents inherit the finding | 15 min |

### Success-criteria gate

Before claiming a win:
- Cosine vs oracle ≥ 0.99 (correctness — `feedback_correctness_first.md`)
- Bench numbers are MEASURED end-to-end, not projected (`feedback_real_vs_projected.md`)
- Sync barriers wrap timing region (`feedback_sync_bounded_timing.md`)
- Probe ran against PRODUCTION-magnitude inputs + multi-position state (`feedback_isolation_must_match_production.md`)
- An external oracle (HF or numpy fp32 — NOT your own ttnn code) is the comparison baseline

---

## 14. Pitfalls (meta-lessons)

These are the recurring failure modes from session history. Treat them as red flags during code review.

### 14.1 Isolation ≠ production (`feedback_isolation_must_match_production.md`)

During Branch III, 5 distinct bugs hid behind cosine-0.997 isolation tests because:
- Test inputs were embed-scale (‖·‖ ≈ 1) but mid-stack hidden states are ‖·‖ ≈ 10-150
- Tests ran 1-2 token chunks; bugs needed 5+ positions of state accumulation
- Reference was the same numpy code path being tested (false positive — both sides shared the bug)

**Fix:** feed HF's actual layer-N hidden states (saved per-layer dumps) as input. Run ≥5 positions. Compare against an EXTERNAL oracle.

### 14.2 Projection ≠ measurement (`feedback_real_vs_projected.md`)

Per-block ms × layer count is a **ceiling**, NOT tok/s. Real tok/s requires full `bench_decode`. Always label "projected" or "ceiling" explicitly. Demo headers, memory notes, and PR descriptions must distinguish:

| Correct framing | Wrong framing |
|---|---|
| "Per-block traced latency: 1.21 ms ✓. End-to-end ms/tok: NOT YET MEASURED — pending C'7.8." | "Multi-chip TP delivers ~16 tok/s (3× faster than single-chip)." |

### 14.3 Isolation probe perf claims <5% don't survive (`feedback_v2_rope_perf_wash.md`)

V2 RoPE isolation projected -8.7% at gated_attn level (-2.18 ms/tok across 16 layers). Production delta: +1.95 ms (within ±2 ms variance — a WASH). Causes:
- Pipelining at full-decode level hides per-layer savings (see `feedback_pipelining_already_wins.md`)
- More dispatched ops can outweigh smaller data sizes

**Rule:** don't trust isolation probe perf claims if projected savings < 5% of full decode. Use isolation for correctness, full-decode for perf.

### 14.4 Cliff hides in defaults (`feedback_fp32_sdpa_cliff_probe.md`)

The 500-position cosine cliff at pos 129 was caused by `compute_kernel_config = HiFi4 + fp32_dest_acc_en=True` — counter-intuitively the "higher-fidelity" setting. Tenstorrent's own Galaxy demo (`models/demos/llama3_70b_galaxy/tt/model_config.py:1202-1207`) uses HiFi2 + `fp32_dest_acc_en=False` for SDPA decode. We were 1.8% slower at HiFi2 but went from 27.8% top-1 at pos 500 to 98.4% top-1. **Compute kernel config flags are not neutral defaults; check what production demos use.**

### 14.5 Async kernel launches fool timing (`feedback_sync_bounded_timing.md`)

`ttnn` op dispatch is async. Naive `time.perf_counter()` around an op measures DISPATCH only, not EXECUTE. Always sync BEFORE start and AFTER stop.

### 14.6 SSH stdout is block-buffered (`feedback_python_stdout_buffering.md`)

SSH pipes have no TTY. Without `sys.stdout.reconfigure(line_buffering=True)` at probe top, logs appear only at process exit — making long probes look hung.

### 14.7 `pkill -9` wedges mesh fabric (`feedback_mesh_recovery_after_kill.md`)

SIGKILL'ing a mesh process corrupts fabric state. Next mesh open hangs. Recover with `ssh qb2 'tt-smi -r 0,1,2,3'`. Use SIGTERM via `serve_tp.sh stop` whenever possible.

### 14.8 ttnn opens ALL chips (`feedback_mtp_head_probe.md`)

`device_ids=[N]` does NOT restrict to one chip. ttnn opens all 4. Two concurrent ttnn processes on the same host SIGBUS each other. If the server is up, no probe can run on that host — coordinate.

### 14.9 `AutoModel.from_pretrained` crashes on remote (`reference_hf_oracle_pattern.md`)

Use the canonical pattern: construct `DecoderLayer` directly + `safe_open` weights. Don't go through `AutoModel`.

### 14.10 Inline `python -c` is banned (`feedback_no_inline_scripts.md`)

Always write a permanent helper in `experiments/utils/`. See `reference_inline_script_helpers.md` for templates (`syntax_check.py`, `ttnn_introspect.py`, `npz_inspect.py`, `hf_download.py`).

### 14.11 Compute kernel config must be all-or-nothing (`feedback_compute_kernel_config.md`)

`WormholeComputeKernelConfig` mixing on Blackhole corrupts ops silently. Either all calls use the same config, or split into a dedicated config object per call site. Don't toggle one flag and reuse.

### 14.12 paged_update_cache trace requirement (`feedback_update_cache_tensor_api_gap.md`)

Our ttnn build's `update_cache_for_token_` has no `cur_pos_tensor=` kwarg (only int `update_index`). Means non-paged path is NOT traceable. Any traced multi-step decode MUST use `paged_update_cache` with `update_idxs_tensor=`. Verified 91f single-chip already does this.

### 14.13 Matmul shape-valid does not mean numerically equivalent

For GDN prediction, the clean formulation:

```text
ttnn.matmul(k4, state_scaled)
[1,12,1,128] @ [1,12,128,128] -> [1,12,1,128]
```

works, but it differs from the existing TTNN broadcast-reduce recurrence by
max diff `0.0625` on real layer-0 seeded tensors. The custom component is much
closer to broadcast-reduce than to matmul. Do not change the correctness
reference to matmul unless you are intentionally changing the numerical
contract and have token-level validation.

### 14.14 GDN debug mode 2 is wedge-prone

`qwen36_gdn_prediction(debug_mode=2)` full strict prediction tied up the qb2
resident server and required killing/restarting it. Modes 10, 11, and 12 are
safe enough to run individually; do not run all debug modes together.

---

## 15. Friend repo (REFERENCE ONLY)

There's a friend's daily-driver Qwen3.6-27B implementation at:
```
experiments/.refs/tt-qwen-36/   (commit a3d12574, branch qwen36-fresh)
```

**Friend (samjett) achieves ~15.3-15.5 tok/s on 4× P150 dense.** Treat this as a reference implementation and pattern catalog, not the optimization target. The target is hardware-ceiling proximity with measured full-decode tok/s and correctness gates.

### WARNING: REFERENCE ONLY

- **Do NOT copy code wholesale.** This is a learning project.
- **Do NOT vendor their patterns into our tree without understanding them.**
- **DO** read it as a reference for "what production-quality multi-chip Tenstorrent inference looks like".
- **DO** cite it as `experiments/.refs/tt-qwen-36/<path>:<line>` when designing.

Comparison table at `research/friend_repo_borrow_list.md`. Key gaps:

| Component | Friend | Ours |
|-----------|--------|------|
| GDN recurrence | Custom C++ `ttnn.experimental.qwen36_gdn_decode` | Manual TTNN broadcast-reduce in production; owned custom GDN exists but is **not default-safe** |
| GDN Q/K/V prep | Custom C++ `qwen36_gdn_prepare_decode` | Python/TTNN slice+reshape+repeat_interleave; not the current blocker |
| RMSNorm | Distributed `rms_norm_pre_all_gather` + `_post_all_gather` | Single fused `ttnn.rms_norm` on replicated x |
| RoPE | `ttnn.experimental.rotary_embedding` + slice/concat | Manual rotate-only |
| LM head | Vocab-sharded, DRAM-sharded multi-split, no final all_reduce | Vocab-sharded + on-device argmax shipped in P22; still has final all_gather |
| Embedding | On-device `ttnn.embedding` | On-device `ttnn.embedding` shipped in P25 |
| Position | On-device `ttnn.plus_one` | Host `cur_pos += 1` |
| Cos/Sin | Precomputed device cache → `ttnn.embedding` | On-device `ttnn.embedding` shipped in P25 |
| Argmax | `all_gather → untilize → ttnn.argmax` | On-device argmax shipped in P22 |

The current custom GDN work should use friend code only for TT-Metal wiring and
dataflow patterns. The friend's recurrence has known/possible errors for our
contract and must not be used as ground truth.

---

## 16. Quick-start checklist (first 30 min as a new agent)

| # | Action | Time |
|---|--------|------|
| 1 | Read `CLAUDE.md` (project root) | 3 min |
| 2 | Read §0 of this `HANDOFF.md` and `research/ACTIVE_CONTEXT.md` | 10 min |
| 3 | Read this `HANDOFF.md` end-to-end | 10 min |
| 4 | Read `MEMORY.md` index if available; otherwise use in-repo `research/` and `experiments/owned_ops/*/README.md` | 8 min |
| 5 | Read `reference_how_to_run_stuff.md` carefully if present | 5 min |
| 6 | Verify hosts are reachable: `ssh qb1 'tt-smi -s'` and `ssh qb2 'tt-smi -s'` | 1 min |
| 7 | Check server status on both hosts: `serve.sh status` / `serve_tp.sh status` or `client_tp status` | 1 min |
| 8 | If server is up on the host you need: use the protocol pattern in §4. If you need raw device: ASK USER before stopping the server. | — |
| 9 | Read 3-4 notes most relevant to the task | varies |
| 10 | Write a plan/design memo with hypothesis, oracle, success gate, and rollback. | 15 min |

### Sanity checks before you write a single line of code

- [ ] You know which host (qb1 / qb2)
- [ ] You know whether the server is up there
- [ ] You have a cited memory note that's relevant
- [ ] You have a cited tt-metal doc path (for non-trivial ttnn work)
- [ ] You have an oracle to compare against (HF or numpy fp32)
- [ ] You have a success criterion in cosine + tok/s terms
- [ ] For GDN work: you understand that `ttnn.matmul(k, state_scaled)` is
      shape-valid but numerically different from the current broadcast-reduce
      recurrence contract

---

## 17. Glossary

| Term | Meaning |
|------|---------|
| **P150** | Tenstorrent Blackhole chip (newer than Wormhole). 4 per host. ~512 GB/s DRAM bandwidth peak. |
| **(1, 4) mesh** | 1-row, 4-column device mesh on qb2 for tensor parallelism |
| **TP** | Tensor parallelism — shard weights across chips, replicate activations |
| **MAX_POS** | KV cache max sequence length (production default 256, probes go up to 32k) |
| **paged SDPA** | `ttnn.transformer.paged_scaled_dot_product_attention_decode` — page-table KV cache layout |
| **B3** | HiFi2 + `fp32_dest_acc_en=False` SDPA compute kernel config — fixes long-context cliff |
| **DeltaNet (DN)** | Linear-attention recurrent block, half of Qwen3.6-27B layers |
| **Gated Attention** | Full-attention block, the other half of Qwen3.6-27B layers (16 of 64) |
| **MLP** | Standard SwiGLU feed-forward |
| **GQA** | Grouped-Query Attention (Qwen3.6: 24 Q heads, 4 KV heads, ratio 6:1) |
| **MTP** | Multi-Token Prediction head (Qwen3.6 ships one; D' probe for speculative decode) |
| **RMSNorm** | Root-mean-square layer norm |
| **rms_norm_pre/post_all_gather** | Distributed RMSNorm — pre computes partial stats, all_gather, post applies norm |
| **all_reduce / all_gather / reduce_scatter** | CCL collectives across mesh chips |
| **CCL** | Collective Communication Library (Tenstorrent's term) |
| **Tracy** | Sampling profiler for device-side op timing. Known configured on qb1; verify qb2 before using for multi-chip. |
| **trace** | Pre-compiled execution graph replayed each token — `ttnn.begin_trace_capture` / `execute_trace` |
| **paged_update_cache** | In-place KV cache write with `update_idxs_tensor` — traceable |
| **L1** | On-chip SRAM (vs DRAM, HBM-like external memory) |
| **bf8 / BFP8** | Tenstorrent block-float-8 weight format |
| **HiFi2 / HiFi4** | Math fidelity flags — counter-intuitively, HiFi2 is correct for SDPA decode on Blackhole, HiFi4 has a bug |
| **fabric** | Inter-chip ethernet/PCIe link — qb1 has none, qb2 has working FABRIC_1D |
| **sub-device** | Galaxy-style core-group partition (prefetcher + worker) — not yet used here |
| **dram_prefetcher** | Streams weights into Global Circular Buffer to hide DRAM load — Galaxy's biggest lever |
| **Agent X/Y/W/O/...** | Sub-agent contributions in 4-agent parallel research pattern |
| **Branch III / C' / C'7 / D'** | Project-internal branch labels for phases of the 27B effort |
