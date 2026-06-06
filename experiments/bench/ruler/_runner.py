#!/usr/bin/env python3
"""RULER driver scaffold — phase-0 stub for next-session wiring.

Documentation-shaped Python entrypoint for driving NVIDIA RULER
(`https://github.com/NVIDIA/RULER`) against our CB OpenAI server
(`experiments/serve/cb_api.py`). No model calls happen here yet.

The phase-1 owner fills in `_invoke_ruler_subprocess` to shell out to
`~/RULER/scripts/run.sh` with the right `OPENAI_API_BASE` env, then
copies the score JSON into our repo-visible results dir.

Compose-from-shelf:
  - `experiments/cb/_runner.py` for `project_root()` + `log()` (existing
    utility shelf — see `feedback_reuse_mandate.md`).
  - Forks the shape of `experiments/cb/load/concurrent_chat.py` which drives
    /v1/chat/completions from a separate process against the running daemon.

Run (after phase-1 wiring):
    .venv/bin/python -m experiments.bench.ruler._runner \\
        --backend gemma4_12b --task-set v0_5_accept_smoke

Until then, this prints the plan and exits.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _project_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "experiments" / "serve").is_dir():
            return p
    raise RuntimeError(f"project root not found from {Path(__file__).resolve()}")


PROJECT = _project_root()
sys.path.insert(0, str(PROJECT / "experiments" / "cb"))
from _runner import log  # noqa: E402 — reuse existing shelf


# Mirrors `experiments/serve/cb_api.py:BACKENDS`. Adding a backend here =
# adding it there too. We don't import that table to avoid pulling ttnn at
# bench-driver time (this script runs OUTSIDE the device process).
BACKENDS = {
    "gemma4_12b":     "google/gemma-4-12b-it",
    "27b":            "Qwen/Qwen3.6-27B",
    "35b":            "Qwen/Qwen3.6-35B-A3B",
    "nemotron3_nano": "nvidia/Nemotron-3-Nano-30B-A3B",
}

# v0.5 accept gate definitions — must stay in lock-step with tasks.yaml.
TASK_SETS = {
    "v0_5_accept_smoke": {
        "tasks":      ["niah_single_1"],
        "lengths":    [4096],
        "samples":    10,
    },
    "v0_5_accept": {
        "tasks":      ["niah_single_1", "niah_multikey_2", "vt"],
        "lengths":    [4096, 8192, 16384, 32768],
        "samples":    50,
    },
    "v0_5_bench_full": {
        "tasks":      ["niah_single_1", "niah_single_2", "niah_single_3",
                       "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
                       "niah_multivalue", "niah_multiquery",
                       "vt", "cwe", "fwe", "qa_1", "qa_2"],
        "lengths":    [4096, 8192, 16384, 32768, 65536, 131072],
        "samples":    500,
    },
}


def _invoke_ruler_subprocess(backend: str, task_set: str, dry_run: bool) -> int:
    """Phase-1 owner fills this in: shell out to `~/RULER/scripts/run.sh`.

    Required env on qb2:
        TT_BACKEND={backend}                    # selects our cb_api backend
        OPENAI_API_BASE=http://localhost:8000/v1
        OPENAI_API_KEY=dummy                    # ignored by cb_api but required by openai SDK

    Subprocess (one shot):
        bash ~/RULER/scripts/run.sh {backend} synthetic

    Per-task overrides (`tasks`, `lengths`, `samples`) are applied by editing
    `~/RULER/scripts/{config_models.sh,config_tasks.sh}` from tasks.yaml
    before invocation, OR by passing CLI overrides to RULER's call_api.py
    directly (preferred — leaves the upstream config untouched).
    """
    log(f"[stub] would run RULER: backend={backend} task_set={task_set} dry_run={dry_run}")
    log(f"[stub] target server: {os.environ.get('OPENAI_API_BASE', 'http://localhost:8000/v1')}")
    log("[stub] phase-1: implement subprocess.run(['bash', '~/RULER/scripts/run.sh', ...])")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--backend", choices=sorted(BACKENDS), default="gemma4_12b")
    p.add_argument("--task-set", choices=sorted(TASK_SETS), default="v0_5_accept_smoke")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan + exit. Default until phase-1 lands.")
    args = p.parse_args()

    plan = TASK_SETS[args.task_set]
    model_id = BACKENDS[args.backend]
    log(f"backend={args.backend} model_id={model_id}")
    log(f"task_set={args.task_set} → {len(plan['tasks'])} tasks × "
        f"{len(plan['lengths'])} lengths × {plan['samples']} samples")
    log(f"tasks={plan['tasks']}")
    log(f"lengths={plan['lengths']}")

    if args.dry_run or os.environ.get("TT_RULER_PHASE", "0") == "0":
        log("phase-0 stub: not invoking RULER. See research/ruler_gemma4_integration_plan.md")
        return 0
    return _invoke_ruler_subprocess(args.backend, args.task_set, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
