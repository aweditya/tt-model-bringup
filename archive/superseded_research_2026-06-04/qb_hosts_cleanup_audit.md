# qb1 / qb2 cleanup audit — 2026-05-30

Read-only audit. No deletions or moves performed. Sizes / mtimes captured live.

---

## Per-host disk summary

### qb1 (tt-qb-ac-01)
- `df -h /`: **692 GB used / 3.6 TB total (20%)**, 2.8 TB free.
- Top 5 home-dir entries by size:
  1. `~/tt-xla/` — **168 GB** (94% is `.cache/`)
  2. `~/tenstorrent/` — **7.1 GB** (5.4 GB is `tt-metal/`)
  3. `~/bonsai_tt_c17/` — **4.0 GB**
  4. `~/tt-xla-fresh-test/` — **1.5 GB**
  5. `~/tt-xla-public-clone/` — **80 MB**
- Plus dotdirs: `~/.cache/huggingface` 143 GB, `~/.cache/uv` 11 GB, `~/.cache/tt-metal-cache` 6.8 GB, `~/.cache/pip` 2.7 GB, `~/.local/lib` 4.7 GB.

### qb2 (tt-qb-ac-02)
- `df -h /`: **402 GB used / 3.6 TB total (12%)**, 3.1 TB free.
- Top 5 home-dir entries by size:
  1. `~/tt-xla/` — **94 GB** (94% is `.cache/`)
  2. `~/tenstorrent/` — **6.3 GB** (4.6 GB is `tt-metal/`)
  3. `~/tt-model-bringup-fresh/` — **4.9 GB** (4.9 GB is `.venv/`)
- Plus dotdirs: `~/.cache/huggingface` 120 GB, `~/.cache/uv` 11 GB, `~/.cache/tt-metal-cache` 4.8 GB.

Disk is NOT under pressure on either host. Both repos `~/tt-xla` are rsync copies (no `.git`), not real clones — git history lives only on the laptop.

---

## DEFINITELY SAFE — old / orphaned dirs

### qb1
| Path | Size | Last mod | Why safe |
|---|---|---|---|
| `~/bonsai_tt_c1/` … `~/bonsai_tt_c20/` (16 dirs) | **4.4 GB total** (c17 alone 4.0 GB; rest 0.4 GB) | May 14–19 | One-shot tiled-kernel staging dirs (`c4_layer0_e2e`, `c11_qlinear_tile`, etc.). No `.git`. Not referenced by any code path in `tt-xla/`. `c17/data/` is a Qwen2.5-0.5B safetensors copy (2.2 GB) already cached in `~/.cache/huggingface`. |
| `~/tt-xla-fresh-test/` | **1.5 GB** | May 21 | `.git` is a worktree pointer to `/Users/adityasriram/Labs/stanford/cs440lx/tt-xla/.git/worktrees/agent-aa634853...` — that path doesn't exist on qb1. Dead worktree. Has its own `.venv` (5 GB). |
| `~/tt-xla-public-clone/` | **80 MB** | May 21 | Duplicate clone of `github.com/aweditya/tt-model-bringup`, untouched since May 21. |
| `~/generated/` | **190 MB** | May 19 | `inspector/` + `watcher/` outputs from May 12–19, not referenced from `tt-xla/`. |
| `~/tt-xla/.cache/hf/` | **110 GB** | (in use) | **DUPLICATE of `~/.cache/huggingface` (143 GB) — different inodes, not a symlink.** Has 8 model dirs that all also exist in `~/.cache/huggingface`. Production now uses `HF_HOME=~/.cache/huggingface`. Confirm no scripts still hardcode `tt-xla/.cache/hf` before removing. |
| `~/tt-xla/.cache/perf_logs/` | **42 GB** | May 24–25 | 15 Tracy capture dirs (`tracy_one_moe_*`, `tracy_traced_decode_v2`, `tracy_v3`, etc.) from the May 24–27 perf session. Each is 2.5–4.6 GB of `.tracy` blobs. Analyses already extracted to `wiki/` and commits. |
| `~/tt-xla/.cache/p21_fp32_sdpa_probe/` | **2.4 GB** | May 14 | 5× 474 MB logits dumps from P21 fp32-SDPA probe (concluded: HiFi4 was the bug; B3 shipped). Result is in MEMORY.md. |
| `~/tt-xla/.cache/qb2_35b_moe/` | **6.3 GB** | May 21 | npz references for the 35B MoE bringup (b0/b1/b2/b3/b3p/b4). Same data also lives on qb2. |
| `~/tt-xla/.cache/sanity_2026_05_22/` | **104 MB** | May 24 | One-off sanity dump. |
| `~/tt-xla/.cache/cosine_ladder_v2/` | **948 MB** | May 14 | Replaced by `cosine_ladder_2026_05_27_*.json` (much smaller). |

**qb1 DEFINITELY SAFE total: ~167 GB** (dominated by `tt-xla/.cache/hf` duplicate + perf_logs).

### qb2
| Path | Size | Last mod | Why safe |
|---|---|---|---|
| `~/tt-xla/.cache/hf/` | **76 GB** | (in use) | **DUPLICATE of `~/.cache/huggingface` (120 GB) — separate inode.** Contains Qwen3.6-27B (52 GB) + Llama variants already in `~/.cache/huggingface`. Same caveat as qb1: confirm no script hardcodes this path. |
| `~/tt-xla/.cache/qb2_tp_deltanet/` | **5.4 GB** | May 19 | TP DeltaNet validation npz dumps from G0–G4 bring-up. Conclusions captured in MEMORY (`feedback_c74_deltanet_tp_pass`, etc.). |
| `~/tt-xla/.cache/qb2_35b_moe/` | **6.3 GB** | May 21 | Same 35B MoE reference dump as qb1 — duplicate. |
| `~/tenstorrent/tt-metal/build_tracy_gcc12/` | **175 MB** | May 15 | Stale earlier Tracy build; current build is `build_tracy_gcc12_nodist/` (1.7 GB) linked as `build`. |
| `~/tenstorrent/tt-metal/build_Release/` | **284 KB** | May 15 | Empty CMakeCache shell — never actually built on qb2 (qb2 uses the tracy variant as `build`). |

**qb2 DEFINITELY SAFE total: ~88 GB.**

---

## SAFE WITH CONFIRMATION

### qb1
| Path | Size | Notes |
|---|---|---|
| `~/.cache/uv/archive-v0/` | **11 GB** | uv package archive. Safe to clear (`uv cache prune`) — will re-download on next `make setup`. |
| `~/.cache/pip/` | **2.7 GB** | Same; safe (`pip cache purge`). Will re-fetch on demand. |
| `~/.cache/ccache/` | **183 MB** | Small; useful if you rebuild tt-metal. Leave. |
| `~/.cache/tt-metal-cache/` | **6.8 GB** | Compiled kernel cache (mainly `7480443082328483327/` 5.5 GB + `10046506135785229659/` 974 MB). Safe to nuke per-key directories you don't recognize, but first decode runs after nuking will be slow as ttnn JIT recompiles. |
| `~/.local/lib/python3.10/` | **4.7 GB** | System pip installs (predates the venv workflow). Verify nothing still imports from here, then can prune. |
| `~/tt-xla/.cache/hf_oracle_35b_*` (5 dirs) | **727 MB** total | May 21–24 needle-haystack HF oracle dumps for L31/L32/L39 probes. The L31/L39 investigation is resolved; keep until you're sure no rerun is needed. |
| `~/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B` (2.4 GB), `Llama-3.2-3B` (6.1 GB), `Meta-Llama-3.1-8B-Instruct` (15 GB) | **23 GB** | Llama weights from earlier experiments. Not used by current Qwen3.6 production. |
| `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B` | **954 MB** | Old smoke-test model. |

### qb2
| Path | Size | Notes |
|---|---|---|
| `~/.cache/uv/` | **11 GB** | Same as qb1 — `uv cache prune` is safe. |
| `~/.cache/pip/` | **29 MB** | Tiny; leave. |
| `~/.cache/ccache/` | **462 MB** | Leave (useful for tt-metal rebuilds). |
| `~/.cache/tt-metal-cache/` | **4.8 GB** | Same caveat as qb1. |
| `~/.cache/huggingface/hub/models--Qwen--Qwen3-Next-80B-A3B-Instruct/` | **16 MB** | Tiny stub (looks like a failed download); safe. |
| `~/tt-model-bringup-fresh/.venv/` | **4.9 GB** | Second venv. If you don't actively use this clone, the venv can be recreated. The clone itself (~50 MB without venv) is fine to keep for fresh-clone sanity runs. |

---

## PROBABLY KEEP — ambiguous

- `~/tt-xla/.cache/cosine_ladder_hf_ref.npz` + `cosine_ladder_tt_logits.npz` (190 MB total, May 14) — referenced by the cosine-ladder workflow; keep until that suite is retired.
- `~/tt-xla/.cache/legacy_demos_2026_05_21/`, `accidental_sync_20260517/` — small (<1 MB each), worth keeping as record.
- `~/tt-xla/.cache/ttnn_so_backups/` (14 MB on qb1), `ttnn_llk_backup/` (1.7 MB on qb2) — backups of patched ttnn `.so` files. Cheap insurance.
- `~/tt-xla/.cache/build/`, `build_logs/`, `runs/`, `c_prime_logs/` — small, runtime artifacts; ignore.

---

## LOAD-BEARING — DO NOT TOUCH

These are explicitly enumerated in the user brief plus what I verified is actively used:

- `~/tt-xla/` (the repo itself, sans `.cache/hf` and sans `.cache/perf_logs`)
- `~/tt-xla/.venv/` (qb1: 5.0 GB, qb2: 5.5 GB)
- `~/tt-xla/.cache/server.log`, `~/tt-xla/.cache/server_tp.log`, PID files — active daemons.
- `~/tenstorrent/tt-metal/` and its `build_Release/` on qb1 (1.4 GB)
- `~/tenstorrent/tt-metal/build_tracy/` on qb1 — **mid-build at audit time** (mtime 15:29, dir created 15:25). Hands off.
- `~/tenstorrent/tt-metal/build_tracy_gcc12_nodist/` on qb2 (1.7 GB) — current Tracy build, linked as `build`.
- `~/tenstorrent/sfpi-gcc/` (1.4 GB each host) — required compiler for tt-metal.
- `~/tenstorrent/{tt-llk,tt-mlir,tt-umd,tt-system-firmware,tt-smi,...}` — small (<120 MB each), all referenced by tt-metal/firmware paths.
- `~/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B` (52 GB) and `models--Qwen--Qwen3.6-35B-A3B` (67 GB on qb1 only) — active production model weights.
- `~/.cache/tt-metal-cache/` — slow to regenerate; keep.
- `~/.local/bin/uv` and other binaries.
- `~/tt-model-bringup-fresh/` on qb2 (sans `.venv`) — used per MEMORY for fresh-clone sanity runs.

---

## Recommended action sequence

Run **on each host**. Sizes are the freed-on-disk estimate.

### Phase 1 — pure duplicates / dead worktrees (≈170 GB on qb1, ≈82 GB on qb2)

First, verify nothing hardcodes the duplicate HF cache path:
```sh
grep -rn "tt-xla/.cache/hf" ~/tt-xla/experiments ~/tt-xla/scripts ~/tt-xla/pjrt_plugin 2>/dev/null
```
If clean (expected — production now uses `HF_HOME=~/.cache/huggingface`), then:

```sh
# qb1
rm -rf ~/tt-xla/.cache/hf            # 110 GB — duplicate HF cache
rm -rf ~/tt-xla/.cache/perf_logs     # 42 GB — old Tracy captures
rm -rf ~/tt-xla-fresh-test           # 1.5 GB — dead worktree pointer
rm -rf ~/tt-xla-public-clone         # 80 MB — duplicate clone
rm -rf ~/generated                   # 190 MB — orphan outputs

# qb2
rm -rf ~/tt-xla/.cache/hf            # 76 GB — duplicate HF cache
rm -rf ~/tenstorrent/tt-metal/build_tracy_gcc12      # 175 MB — stale tracy build
rm -rf ~/tenstorrent/tt-metal/build_Release         # 284 KB — empty shell on qb2 only
```

### Phase 2 — old experiment dirs (≈18 GB on qb1)

```sh
# qb1
rm -rf ~/bonsai_tt_c{1,2,4,5,6,7,8,9,10,11,15,16,17,18,19,20}   # 4.4 GB total
rm -rf ~/tt-xla/.cache/p21_fp32_sdpa_probe                       # 2.4 GB
rm -rf ~/tt-xla/.cache/qb2_35b_moe                               # 6.3 GB (also on qb2)
rm -rf ~/tt-xla/.cache/cosine_ladder_v2                          # 948 MB
rm -rf ~/tt-xla/.cache/sanity_2026_05_22                         # 104 MB
rm -rf ~/tt-xla/.cache/hf_oracle_35b_needle100_L{31,32,39,39_v2} # ~600 MB
# qb2
rm -rf ~/tt-xla/.cache/qb2_tp_deltanet                           # 5.4 GB
rm -rf ~/tt-xla/.cache/qb2_35b_moe                               # 6.3 GB
```

### Phase 3 — package caches (≈14 GB on qb1, ≈11 GB on qb2)

```sh
uv cache prune          # both hosts
pip cache purge         # qb1 only (qb2 pip cache is 29 MB)
```

### Phase 4 — old HF model weights (≈24 GB on qb1, if you're sure)

Only on qb1:
```sh
rm -rf ~/.cache/huggingface/hub/models--unsloth--*               # 23 GB
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B       # 954 MB
```

**Total potential reclaim: qb1 ~220 GB, qb2 ~100 GB.** Neither host is space-constrained, so phases 1+2 alone are sufficient to remove the visual clutter the user mentioned.
