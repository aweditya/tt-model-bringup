# qb2 Remote Cleanup Audit — 2026-05-19

Compiled from a read-only directory inspection of qb2 home + project tree.
**No actions executed.** All commands listed below are for the user to review
and run themselves.

## Summary

qb2 has 226 GB used of 3.6 TB (7% — healthy). `~/tt-xla` is 63 GB, dominated by
`.cache/hf/hub` (52 GB Qwen3.6-27B HF snapshot, KEEP) and
`.cache/qb2_tp_deltanet` (~5.4 GB of cosine-ladder npz dumps, mostly redundant).
**True reclaimable space: ~6.5 GB** from stale npz artifacts + 175 MB optional
from an unused tt-metal build dir.

## Safe to delete (~6.5 GB total)

Old probe npz dumps superseded by newer runs. The active latest is
`_cosine_ladder_tp_owned_gdn_20260519_1909.npz` (May 19).

| Path | Size | Reason |
|---|---|---|
| `~/tt-xla/.cache/qb2_tp_deltanet/_cosine_ladder_tp_manual_20260518_*.npz` (4 files) | ~950 MB | May-18 baselines superseded |
| `~/tt-xla/.cache/qb2_tp_deltanet/_cosine_ladder_tp_manual_20260519_07*.npz` (2 files) | ~125 MB | Older same-day runs |
| `~/tt-xla/.cache/qb2_tp_deltanet/_cosine_ladder_tp_owned_gdn_20260518_*.npz` (2 files) | ~664 MB | Superseded May-18 owned-GDN dumps |
| `~/tt-xla/.cache/qb2_tp_deltanet/_cosine_ladder_tp_owned_gdn_20260519_{06,07}*.npz` (4 files) | ~1.9 GB | Earlier May-19 owned-GDN runs |
| `~/tt-xla/.cache/qb2_tp_deltanet/_cosine_ladder_tp_owned_gdn_20260519_1814.npz` | 474 MB | Superseded by 1905/1909 runs |
| `~/tt-xla/.cache/c_prime_logs/*.log` (8 files) | 152 KB | C'1/C'2 logs, archived in memory |
| `~/tt-xla/.cache/runs/*.log` (15 files) | 148 KB | One-shot probe logs, May-13 |
| `~/tt-xla/.cache/ttnn_llk_backup/` (3 dirs) | 1.7 MB | LLK source backups, May-12 |
| `~/tt-xla/.cache/server.{log,pid,pid.launch,sock}` (May-13 stale) | <1 KB | Orphan from pre-TP `server.py` |
| `~/tt-xla/.cache/hf_layer*_substeps.npz`, `ttnn_layer*_substeps_full.npz`, `hf_per_layer_hidden_states.npz` | ~50 MB | C'0 debug dumps, May-12 |
| `~/tt-xla/.cache/perf_baseline_C*.json`, `b8/b9_diagnostics.json`, `c4_traced_results.json`, etc | ~16 KB | C-prime diagnostics, results in memory |

## Maybe stale — confirm before delete

| Path | Size | Note |
|---|---|---|
| `~/tt-xla/.cache/p18_*`, `p19_*`, `p22_*`, `p24_*`, `p25_*` | ~52 KB | Tiny, findings in memory; likely safe |
| `~/tt-xla/.cache/qb2_tp_{profile,collectives,components,fused_cache,generate_bench,rope,tracy}/` | ~660 KB | Mostly May-14-17 profile JSONs |
| `~/tt-xla/.cache/accidental_sync_20260517/` (3 MD files) | 68 KB | Name suggests accidental — read first |
| `~/tt-xla/.cache/qwen05b/`, `runs/`, `ttnn/` | ~150 KB | Old experiments |
| `~/tenstorrent/tt-metal/build_tracy_gcc12/` | 175 MB | Older Tracy build; active is `build_tracy_gcc12_nodist` |
| `~/tenstorrent/tt-metal/build_Release/` | 284 KB | Likely stub |

## Active — DO NOT TOUCH

- `~/tt-xla/.cache/hf/` (52 GB) — Qwen3.6-27B HF snapshot. Mandatory.
- `~/tt-xla/.cache/server_tp.{log,pid,pid.launch,sock}` — currently-running TP server.
- `~/tt-xla/.cache/qb2_tp_deltanet/_cosine_ladder_tp_owned_gdn_20260519_{1905,1909}.npz` — keep latest two as safety.
- `~/tt-xla/.cache/qb2_tp_deltanet/*.json` (6 files) — current probe summary JSONs.
- `~/tt-xla/.venv/` (5.5 GB) — active Python env.
- `~/tenstorrent/tt-metal/build_tracy_gcc12_nodist/` (1.7 GB) — active build (symlinked as `build`).
- `~/tt-xla/experiments/owned_ops/` (9 owned op dirs) — active custom-op work.
- All `experiments/91*.py` and `experiments/*.py` — historical record, small (<3 MB total).

## Reclaimable commands (user to execute)

```bash
# 1. Stale TP deltanet npz dumps (~6.4 GB) — keep only the two latest
ssh qb2 'cd ~/tt-xla/.cache/qb2_tp_deltanet && \
  ls _cosine_ladder_tp_*.npz | grep -v "20260519_1909\|20260519_1905" | xargs -r rm -v'

# 2. Old non-TP server stragglers (server.py predecessor)
ssh qb2 'cd ~/tt-xla/.cache && rm -v server.log server.pid server.pid.launch server.sock mock_test.log'

# 3. C-prime debug dumps May-12-13
ssh qb2 'cd ~/tt-xla/.cache && \
  rm -v hf_layer0_substeps.npz hf_layer2_substeps.npz hf_per_layer_hidden_states.npz \
        ttnn_layer2_substeps_full.npz perf_baseline_C*.json \
        b8_diagnostics.json b9_diagnostics.json c4_traced_results.json per_layer_diff_results.json \
        kernel_profile_*.json hf_oracle_topk.json tp_decompose_run.log p19_run.log \
        p22_vocab_sharded_lm_head_probe.log qwen36_download.log qwen36_27b_hf_layer0_ref.npz \
        qwen36_27b_layers0_3_ref.npz qwen36_27b_layers0_7_ref.npz'
ssh qb2 'rm -rfv ~/tt-xla/.cache/c_prime_logs ~/tt-xla/.cache/runs ~/tt-xla/.cache/ttnn_llk_backup'

# 4. Empty/stub dirs
ssh qb2 'rmdir -v ~/tt-xla/.cache/build ~/tt-xla/.cache/builds ~/tt-xla/.cache/ttnn-tmp ~/tt-xla/.cache/ttnn 2>/dev/null'

# 5. (OPTIONAL — verify first) Old non-active Tracy build
ssh qb2 'rm -rfv ~/tenstorrent/tt-metal/build_tracy_gcc12 ~/tenstorrent/tt-metal/build_Release'

# 6. (REVIEW FIRST) accidental_sync_20260517 dir — read MD files first
ssh qb2 'ls -la ~/tt-xla/.cache/accidental_sync_20260517/'
```

## Caveats

- TP server is currently running. Do NOT touch `server_tp.{log,pid,pid.launch,sock}` or kill the server PID.
- `~/tt-metal-tracy/` referenced in memory note `reference_tracy_build_qb1.md` does NOT exist on qb2; that note refers to qb1 only. Tracy on qb2 lives at `~/tenstorrent/tt-metal/build_tracy_gcc12_nodist/`.
- `/tmp` has nothing of ours; stale `torchinductor_aditya` dir is system-shared.
