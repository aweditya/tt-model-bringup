#!/usr/bin/env python3
"""v2 wire-up smoke — confirms TT_BACKEND=gemma4_12b dispatches the right
modules in `cb_api.BACKENDS` and `cb_scheduler._BACKEND_MODULES` without
running a full bootstrap. Catches typos + import-time errors fast.

Run (qb1, no harness needed):
  cd ~/tt-xla && TT_BACKEND=gemma4_12b .venv/bin/python -u \\
    experiments/cb/isolate/gm4_v2_wireup_smoke.py
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))
# openai_endpoint imports `experiments.serve.protocol`, so the project
# root must be on sys.path for that namespace package to resolve.
sys.path.insert(0, str(PROJECT_ROOT))

# Force the right backend before importing cb_api (the registry is read at
# module load time).
os.environ.setdefault("TT_BACKEND", "gemma4_12b")
# Skip the openai-app build inside cb_api (no fastapi needed for the smoke).
os.environ.setdefault("TT_OPENAI_BUILD_APP", "0")


def log(msg):
    print(msg, flush=True)


def main():
    log(f"TT_BACKEND={os.environ['TT_BACKEND']}")

    # 1) cb_api registry.
    import cb_api  # noqa: E402
    assert cb_api.TT_BACKEND == "gemma4_12b", \
        f"cb_api.TT_BACKEND={cb_api.TT_BACKEND!r}"
    assert cb_api._BACKEND_MODULE == "server_gemma4_unified_ttnn"
    assert cb_api._BACKEND_DEFAULT_MODEL == "google/gemma-4-12B"
    assert cb_api.DEFAULT_MODEL_ID == "google/gemma-4-12B"
    log(f"  cb_api: backend module={cb_api._BACKEND_MODULE!r}, "
        f"default model={cb_api.DEFAULT_MODEL_ID!r}  PASS")

    # 2) cb_scheduler registry + dynamic import of both base + cb.
    import cb_scheduler  # noqa: E402
    assert cb_scheduler._TT_BACKEND == "gemma4_12b"
    assert cb_scheduler._base_mod == "server_gemma4_unified_ttnn"
    assert cb_scheduler._cb_mod == "server_gemma4_unified_cb"
    log(f"  cb_scheduler: base={cb_scheduler._base_mod!r}, "
        f"cb={cb_scheduler._cb_mod!r}  PASS")

    # 3) The CB module exposes the scheduler-required entry points.
    cb = importlib.import_module(cb_scheduler._cb_mod)
    required = ["setup_cb_state", "cb_reset_states", "cb_reset_slots",
                "update_input_buffers_batched", "forward_batch_tp_inner",
                "step_forward_cb"]
    missing = [name for name in required if not hasattr(cb, name)]
    assert not missing, f"cb module missing: {missing}"
    log(f"  cb module exposes: {required}  PASS")

    # 4) The base module exposes the scheduler-required entry points
    # (greedy path; chunked prefill not required for default smoke).
    base = importlib.import_module(cb_scheduler._base_mod)
    required_base = ["State", "bootstrap"]
    missing_base = [name for name in required_base if not hasattr(base, name)]
    assert not missing_base, f"base module missing: {missing_base}"
    log(f"  base module exposes: {required_base}  PASS")

    log("=" * 60)
    log("v2 wire-up smoke: ALL PASS — ready for full HTTP smoke")
    log("=" * 60)


if __name__ == "__main__":
    main()
