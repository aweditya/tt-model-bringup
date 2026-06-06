"""Round 10 ablation — revert MLP weights back to default (interleaved DRAM)
in the live dev-harness state so we can A/B vs the DRAM-sharded variant
without a 14-min cold restart.

Forks `gm4_round10_dram_mlp_revalidate.py` (sister script): same re-upload
loop, just sets TT_GM4_DRAM_PREFETCH=0 before the reload so the new
upload uses the default INTERLEAVED-DRAM memcfg. Used to confirm the
n=3 baseline traced ms/tok in the SAME process where the DRAM-sharded
n=3 was measured (eliminates any thermal / driver-state confound).

Trigger:
  ssh qb2 'touch tt-xla/.cache/gm4_runtime/trig/round10_revert_to_default_mlp'
"""
from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import server_gemma4_unified_ttnn as srv  # noqa: E402

# Reuse the helper from the sister script (single source of truth for the
# safe re-upload pattern).
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "cb" / "isolate"))
import gm4_round10_dram_mlp_revalidate as sib  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(state=None):
    if state is None:
        log("ERROR: needs dev harness (resident State)")
        return 1

    os.environ["TT_GM4_DRAM_PREFETCH"] = "0"
    log("set TT_GM4_DRAM_PREFETCH=0 (revert to default INTERLEAVED-DRAM)")

    importlib.reload(srv)
    srv_mod = sys.modules["server_gemma4_unified_ttnn"]
    log(f"server module reloaded; _dram_sharded_enabled() = {srv_mod._dram_sharded_enabled()}")

    log("re-uploading all 48 MLP layers under default path…")
    sib._reupload_all_mlp(state, srv_mod, log=log)

    log("invalidating cached trace…")
    sib._invalidate_trace(state, log=log)

    log("running v04 validator under default MLP path…")
    result = sib._run_validator(state, srv_mod, label="round10-default", log=log)

    log("=" * 78)
    verdict = "PASS" if result["match"] == result["total"] else "FAIL"
    log(f"VERDICT: {verdict}  (match {result['match']}/{result['total']}, "
        f"eager {result['eager_ms']:.1f}, traced {result['traced_ms']:.1f} ms/tok)")
    log("=" * 78)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
