# qb1 Bootstrap Plan (Revised)

## Context

qb1 (`ssh qb1`) replaces the disconnected `ssh tenstorrent` host. Kernel driver loaded, `/dev/tenstorrent/{0,1,2,3}` present (4 chips, we use device 0 only). Otherwise fresh: no ttnn, no jax, no tt-metal for our user. Bootstrap from scratch.

## Decisions (locked)

- **Package manager: uv** (Astral) — fast, deterministic, single binary install
- **tt-metal pin: `v0.69.0`** — latest stable release (May 4, 2026); `v0.70.0` only has -rc tags, not stable
- **Repo list: everything potentially useful** (see Phase B) — cheap to clone, expensive to need later and not have
- **Device: 0** (saturate first, then expand)
- **Environment: all local** to project — venv at `~/tt-xla/.venv`, caches at `~/tt-xla/.cache/`
- **No `/tmp` usage anywhere**

## Directory Layout on qb1

```
/home/aditya/
├── tt-xla/                    # our project (rsync from local)
│   ├── .venv/                 # uv-managed project venv
│   ├── .cache/                # all caches: build logs, model cache, ttnn cache
│   │   ├── build.log
│   │   └── ttnn/              # TT_METAL_HOME-scoped cache, NOT /tmp
│   └── ... (existing tree)
└── tenstorrent/               # cloned reference + source repos
    ├── tt-metal/              # pin v0.69.0; provides ttnn (heavy build)
    ├── tt-mlir/               # official PJRT/MLIR compiler — reference
    ├── tt-llk/                # low-level kernels (separate from metal's vendored copy)
    ├── tt-lang/                # Python DSL for ops
    ├── tt-isa-documentation/  # ISA docs
    ├── tt-umd/                # user-mode driver
    ├── tt-smi/                # CLI hardware tool
    ├── tt-exalens/            # hw debugger
    ├── tt-forge-models/       # model implementations
    ├── tt-vscode-toolkit/     # VSCode tools
    ├── tt-system-firmware/    # firmware reference
    ├── tt-buda/               # older stack — reference
    ├── tt-inference-server/   # serving reference
    ├── tt-xla-official/       # official PJRT plugin (named tt-xla on github) — for comparison
    ├── sfpi/                  # SFPI library
    ├── sfpi-gcc/              # GCC fork
    └── polaris/               # platform
```

The official `tt-xla` repo on github is renamed `tt-xla-official` on disk to avoid colliding with our project at `~/tt-xla`.

## Phases

### Phase A — Filesystem prep (~10 sec)

Script: `pjrt_plugin/scripts/qb1_phase_a_prep.sh`
- `mkdir -p ~/tenstorrent ~/tt-xla/.cache/ttnn`
- Set `TTNN_CACHE_DIR=~/tt-xla/.cache/ttnn` in `~/.bashrc` (replaces default `/tmp/ttnn`)

### Phase B — Clone Tenstorrent repos (~10 min over network)

Script: `pjrt_plugin/scripts/qb1_phase_b_clone.sh`

Two batches:
1. **Heavy with submodules** — tt-metal v0.69.0, tt-mlir, tt-llk (clone with `--recursive`)
2. **Light, no submodules** — everything else (clone shallow `--depth=1`)

Parallel where possible (background `&` and `wait`).

### Phase C — Install uv + set up venv (~2 min)

Script: `pjrt_plugin/scripts/qb1_phase_c_venv.sh`
- Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh` (to `~/.local/bin/uv`)
- `uv venv ~/tt-xla/.venv --python 3.10`
- `uv pip install --python ~/tt-xla/.venv/bin/python numpy torch pytest jax==0.6.2 jaxlib==0.6.2`

Note: JAX 0.6.2 may pin specific jaxlib — if 0.6.2 unavailable for py3.10, use the most recent jax that ships PJRT API v0.70.

### Phase D — Build tt-metal (~30-60 min, LONG)

Script: `pjrt_plugin/scripts/qb1_phase_d_build_metal.sh`
- `cd ~/tenstorrent/tt-metal && ./build_metal.sh 2>&1 | tee ~/tt-xla/.cache/build_metal.log`
- After build: tt-metal/python_env contains ttnn install. We add `~/tenstorrent/tt-metal` to PYTHONPATH in our venv via a `.pth` file or wrapper.

This runs in **background**. We continue Phase E in parallel.

### Phase E — Sync our code (~30 sec)

Script: `pjrt_plugin/scripts/qb1_phase_e_sync.sh` (runs from local)
- rsync `pjrt_plugin/`, `research/`, `wiki/`, `CLAUDE.md`, etc. to `qb1:~/tt-xla/`
- Exclude: `.venv/`, `.cache/`, `__pycache__/`, `*.so`

### Phase F — Build PJRT plugin .so (~3 min, after Phase D)

Script: `pjrt_plugin/scripts/qb1_phase_f_build_plugin.sh`
- Source the venv
- Run existing `pjrt_plugin/scripts/build.sh` (fetches pjrt_c_api.h, runs cmake)
- Produces `pjrt_tt.so`

### Phase G — Smoke test (~30 sec)

```
TT_PJRT_USE_DEVICE=1 \
TT_METAL_HOME=~/tenstorrent/tt-metal \
TTNN_CACHE_DIR=~/tt-xla/.cache/ttnn \
~/tt-xla/.venv/bin/python pjrt_plugin/scripts/smoke_test_device.py
```

### Phase H — Full device test suite (~5 min)

```
TT_PJRT_USE_DEVICE=1 \
TT_METAL_HOME=~/tenstorrent/tt-metal \
~/tt-xla/.venv/bin/python -m pytest pjrt_plugin/tests/test_engine_device.py -v
```

## What we will NOT do

- No `/tmp` writes — overridden via `TTNN_CACHE_DIR`, `TT_METAL_CACHE_DIR`, etc.
- No system-wide pip installs — all in project venv
- No touching `/opt/ttmlir-toolchain/` — owned by another user
- No inline scripts — every step is a permanent file in `pjrt_plugin/scripts/`

## Risks & Mitigations

1. **tt-metal v0.69.0 build fails on Blackhole.** Mitigation: log to `.cache/build_metal.log`; fall back to a known-good SHA if needed.
2. **JAX 0.6.2 not available for py3.10.** Mitigation: check `pip index versions jax`; if missing, find closest version that ships PJRT API v0.70.
3. **uv installer needs `~/.local/bin` in PATH.** Mitigation: export PATH in scripts; avoid the bashrc-modifying installer side effect.
4. **Build dependencies missing.** Mitigation: tt-metal install docs list deps; resolve incrementally.

## Verification

After Phase G:
- `jax.devices()` shows TT Blackhole
- `ttnn.open_device(device_id=0)` succeeds
- Round-trip a tensor through `engine._to_device` / `engine._from_device`
- `ttnn.add` runs on device

After Phase H: all device-mode engine tests green.
