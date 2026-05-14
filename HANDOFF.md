# HANDOFF.md — TT-XLA Qwen3.6-27B Agent Onboarding

Last updated: 2026-05-14. Authoritative entry point for any agent picking up this project. Read this end-to-end before touching code. Cite memory notes by filename (e.g. `feedback_paged_sdpa_shipped_tp.md`) — they live in `~/.claude/projects/-Users-adityasriram-Labs-stanford-cs440lx-tt-xla/memory/`.

## Table of contents
1. What this project is
2. Non-negotiables
3. Hosts (qb1 vs qb2)
4. Persistent servers (lifecycle + endpoints)
5. Repo layout
6. Memory bank
7. Roofline / ceiling math
8. Roadmap + shipped wins
9. Reproducing key results
10. Tracing / profiling
11. Web scraping / research strategy
12. Background research strategy (4-agent parallel)
13. Plan-of-action template
14. Pitfalls (meta-lessons)
15. Friend repo (reference only)
16. Quick-start checklist
17. Glossary

---

## 1. What this project is

Stanford **CS440LX exploratory research**: build deep understanding of Tenstorrent Blackhole P150 hardware + JAX/XLA internals, then ship a JAX/XLA backend for Tenstorrent. Active line of work is bringing up **Qwen3.6-27B** (MoE-free hybrid recurrent: DeltaNet + Gated Attention + dense MLP, 64 layers, ~27B params) on **4× P150 with tensor parallelism**. The 27B work is correctness-validated single-chip and now multi-chip; see section 8 for the roadmap and shipped wins.

The 27B bring-up is the proving ground — once correctness and perf are saturated, learnings funnel back into the PJRT plugin (`pjrt_plugin/`) that will be the actual JAX backend.

---

## 2. Non-negotiables

These come from `CLAUDE.md` (project root) and the auto-memory `feedback_non_negotiables.md`. Hard rules — violating them silently wastes hours.

| # | Rule | Rationale |
|---|---|---|
| 1 | **Plan first, act later** | Every claim grounded in experiment. No hand-wavy code. |
| 2 | **Research-driven workflow** | Most output is Q&A, wiki entries, design memos. Code follows. |
| 3 | **No code bloat** | Think more, type less. Concise + correct. |
| 4 | **Remote execution only** — `ssh qb1` / `ssh qb2` | The local Mac has no Tenstorrent hardware. `ssh tenstorrent` (legacy) is GONE. |
| 5 | **Single device by default** | Both hosts have 4 chips. Stick to 1 until saturated unless multi-chip is a real-need decision (memory or fabric requirement). |
| 6 | **No inline scripts** (`python -c '...'`) | Always write a permanent helper in `experiments/utils/` and run that. See `reference_inline_script_helpers.md`. |
| 7 | **No `/tmp`** | All scratch goes in project dirs — `.cache/`, `experiments/`, `research/`, `wiki/`. |
| 8 | **Frequent commits** | Auto-commit allowed (`feedback_commits.md`). Watchdog kills sessions silent > 600s. |
| 9 | **No local execution of device code** | Local Mac is read/write/edit + git only. No `ttnn` imports locally. |
| 10 | **Correctness first** | If cosine < 0.99, STOP and ablate (`feedback_correctness_first.md`). Don't optimize broken math. |
| 11 | **Numpy oracle, not HuggingFace AutoModel** | `AutoModel.from_pretrained` crashes on remote; build pure-numpy fp32 reference + construct `DecoderLayer` directly with `safe_open` weights (`reference_hf_oracle_pattern.md`). |
| 12 | **Sync-bounded timing** | ttnn dispatch is async. `ttnn.synchronize_device(device)` BEFORE start AND AFTER stop, every benchmark (`feedback_sync_bounded_timing.md`). |
| 13 | **Stdout line-buffering for SSH probes** | SSH pipes are block-buffered. Add `sys.stdout.reconfigure(line_buffering=True)` at top of every helper (`feedback_python_stdout_buffering.md`). |
| 14 | **Doc-first** | Before any non-trivial ttnn/mesh/fabric work, read `experiments/.refs/tt-metal/tech_reports/` + `models/demos/llama3_70b_galaxy/`. Cite doc path in design (`feedback_consult_docs_before_acting.md`). |
| 15 | **Never cite projection as measurement** | Per-block × layer count is a CEILING not tok/s. Real numbers require full `bench_decode` (`feedback_real_vs_projected.md`). |

---

## 3. Hosts

Two hosts, both active concurrently. Each has 4× Tenstorrent Blackhole P150. **There are NO NVIDIA GPUs.** Check with `tt-smi -s` (`-s` snapshot flag is required; tt-smi defaults to interactive TUI which hangs SSH).

| Host | Chips | Fabric | Use case | Bootstrap time |
|------|-------|--------|----------|----------------|
| `qb1` | 4× P150 | NO inter-chip fabric | **Single-chip only.** Server `server.py`. Long-context drift work, ttnn op probes, Tracy profiling. | ~11 min weight load |
| `qb2` | 4× P150 | Working fabric | **Multi-chip TP.** Server `server_tp.py`. (1,4) mesh, all_gather/all_reduce, distributed layers. | ~5-10 min sharded load |

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

qb2 (`server_tp.py`) is a SUBSET focused on multi-chip TP — at minimum `handle_generate_tp`. Verify with `grep -n "^def handle_" experiments/serve/server_tp.py`.

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
│   ├── tt_jax/               # tt-metal -> JAX integration scaffolding
│   ├── .refs/                # vendored tt-metal references (read-only)
│   └── logs/
├── demos/                    # presentation demos
└── tt_docs_corpus/           # scraped tt-metal / tenstorrent docs
```

Current focus: `experiments/91*` series + `experiments/serve/server*.py`. The single most-touched production file is **`experiments/serve/server_tp.py`** (qb2 multi-chip).

---

## 6. Memory bank

Auto-memory lives at:
```
~/.claude/projects/-Users-adityasriram-Labs-stanford-cs440lx-tt-xla/memory/
```

The index is `MEMORY.md`. ~140 entries as of 2026-05-14, organized by theme:

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
| Friend's daily-driver | 15.3 tok/s | competitive target |
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

The remaining lever set is the **multi-chip TP opt menu** (`research/multi_chip_optimizations_menu.md` + `_v2_addendum.md`). Top items: distributed RMSNorm (~18 ms/tok), all_gather_concat (2-5 ms/tok), vocab-sharded lm_head (skips final logits AG).

---

