"""Shared bootstrap + log helpers for CB validators / benches / load / profile.

Removes ~10 lines of identical setup boilerplate from each test file:
project-root discovery, sys.path insert, `log(msg)`, the
`MeshServerState() if hasattr(...) else State()` cargo-cult fallback (the
`base.State` path doesn't exist any more), and the canonical CB DN-mode
overrides.

Usage from any test under experiments/cb/**:

    from _runner import bootstrap_27b_cb, log
    state, base = bootstrap_27b_cb()
    log("ready")
    # ... your test body ...

The caller is expected to have added `experiments/cb/` to sys.path; the
established pattern in the test files is to walk parents for the project
root and insert `experiments/serve/` and `experiments/cb/` once at the top
of the file.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path


def project_root() -> Path:
    """Walk parents from this file's location until experiments/serve/ appears."""
    for p in Path(__file__).resolve().parents:
        if (p / "experiments" / "serve").is_dir():
            return p
    raise RuntimeError("project root not found from " + str(Path(__file__).resolve()))


def log(msg) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bootstrap_27b_cb():
    """Bootstrap server_tp and apply the canonical CB DN-mode overrides
    (manual recurrence + manual decay-gate + native_softplus decay — the
    config every CB test in the tree uses).

    Returns `(state, base)` so the caller can poke device tensors and use the
    base module's helpers. Line-buffers stdout/stderr for SSH pipes."""
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    root = project_root()
    serve_dir = str(root / "experiments" / "serve")
    if serve_dir not in sys.path:
        sys.path.insert(0, serve_dir)

    import server_tp as base
    state = base.MeshServerState()
    base.bootstrap(state)
    state.deltanet_recurrence_mode = "manual"
    state.deltanet_decay_gate_mode = "manual"
    state.deltanet_decay_mode = "native_softplus"
    return state, base
