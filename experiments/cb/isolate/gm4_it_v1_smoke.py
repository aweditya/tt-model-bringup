#!/usr/bin/env python3
"""Gemma 4 12B IT smoke — runs v0.3.1 multi-step + v0.3.2 free-run +
v1.6 B=4 gates against the IT variant. Reuses the existing infra
(server module accepts TT_GEMMA4_VARIANT=it; HF oracle at
.cache/hf_oracle_gemma4_12B_it).

Per the model_bringup_recipe.md REUSE mandate: this is a thin wrapper
that calls the existing v0.3.1, v0.3.2, v1.6 main()s with the IT
oracle path. No new forward code.

Run via harness (with TT_GEMMA4_VARIANT=it set BEFORE harness boot):
  TT_GEMMA4_VARIANT=it bash scripts/run_harness_tmux.sh gm4
  ssh qb1 'touch ~/tt-xla/.cache/gm4_runtime/trig/it_v1_smoke'
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "cb" / "isolate"))

import server_gemma4_unified_ttnn as base  # noqa: E402
import server_gemma4_unified_cb as cb      # noqa: E402

# Point the v0.3.1 / v0.3.2 / v1.6 probes at the IT oracle by importing
# them and rewriting their module-level ORACLE_DIR before calling main().
import gm4_v031_multistep_cos as p031  # noqa: E402
import gm4_v032_freerun as p032        # noqa: E402
import gm4_v1_6_b4 as p166             # noqa: E402

IT_ORACLE = PROJECT_ROOT / ".cache" / "hf_oracle_gemma4_12B_it"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(state=None):
    if state is None:
        log("bootstrapping IT (~80s)…")
        state = base.State(); base.bootstrap(state, log=log)
    else:
        log(f"using harness state (variant={getattr(state, 'variant', '?')}, "
            f"model={getattr(state, 'hf_model_id', '?')})")

    if not IT_ORACLE.exists():
        log(f"FATAL: IT oracle missing at {IT_ORACLE}. "
            f"Generate with TT_GEMMA4_MODEL_ID=google/gemma-4-12B-it "
            f".venv-gemma4/bin/python experiments/utils/hf_reference_gemma4_12b.py "
            f"--output-dir .cache/hf_oracle_gemma4_12B_it")
        return 1

    p031.ORACLE_DIR = IT_ORACLE
    p032.ORACLE_DIR = IT_ORACLE

    log("=" * 78)
    log("IT v0.3.1 multi-step (vs IT HF oracle)…")
    log("=" * 78)
    rc_031 = p031.main(state=state)

    log("=" * 78)
    log("IT v0.3.2 free-run (16 tokens after teacher-forced prefill)…")
    log("=" * 78)
    p032.main(state=state)

    log("=" * 78)
    log("IT v1.6 B=4 acceptance gate…")
    log("=" * 78)
    rc_166 = p166.main(state=state)

    verdict = (rc_031 == 0) and (rc_166 == 0)
    log("=" * 78)
    log(f"VERDICT: {'PASS' if verdict else 'FAIL'} "
        f"(v0.3.1 rc={rc_031}, v1.6 B=4 rc={rc_166})")
    log("=" * 78)
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
