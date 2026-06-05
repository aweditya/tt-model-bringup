"""Interactive dev harness for Nemotron-3 Nano — bootstrap once, run many tests.

The Nemotron-3 all-resident bootstrap is ~108s (v0.3.0 measurement).
Without this harness every fix-test cycle pays that cost. Here we
bootstrap once into a long-lived python process and run tests on
demand via trigger files. Iteration becomes:

  EDIT (local) → bash scripts/deploy.sh <files>
  ssh qb1 'touch ~/tt-xla/.cache/nm3_runtime/trig/<test_name>'
  ssh qb1 'cat ~/tt-xla/.cache/nm3_runtime/trig/last.log'

Each trigger:
  1. `importlib.reload`s `server_nemotron3_nano_ttnn` so new code lands
     without re-bootstrap.
  2. Calls `test_module.main(state=state)` — tests must accept a
     pre-bootstrapped state to skip bootstrap.
  3. Catches all exceptions; harness survives bad tests.

REUSE: forks `cb35_dev_harness.py` verbatim with `base = server_nemotron3_nano_ttnn`
and test-name candidates `nm3_<name> | nemotron3_<name> | <name>`.

Bootstrap default: NEMOTRON3_UPLOAD_LAYERS=all (resident path) +
NEMOTRON3_MOE_MODE=ep. Override via env BEFORE launching the tmux session.

Usage:
  # one-time on qb1 (eats ~108s bootstrap, then idles):
  bash scripts/run_harness_tmux.sh nm3

  # per iteration (locally):
  bash scripts/deploy.sh experiments/serve/server_nemotron3_nano_ttnn.py \
                        experiments/cb/isolate/nemotron3_v030_resident_smoke.py
  ssh qb1 'touch ~/tt-xla/.cache/nm3_runtime/trig/v030_resident_smoke'
  ssh qb1 'cat ~/tt-xla/.cache/nm3_runtime/trig/last.log'
"""
from __future__ import annotations

import importlib
import os
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents
                    if (p / "experiments" / "serve").is_dir())
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Default to the resident path; user env can override.
os.environ.setdefault("NEMOTRON3_UPLOAD_LAYERS", "all")
os.environ.setdefault("NEMOTRON3_MOE_MODE", "ep")

import server_nemotron3_nano_ttnn as base  # noqa: E402

TEST_SEARCH_DIRS = ["experiments/cb/validate", "experiments/cb/isolate",
                    "experiments/cb/bench", "experiments/cb/dev"]


def _discover_test_module(trigger_name: str):
    """Return the (re)loaded module for a trigger name, or None.
    Candidates: nm3_<name> | nemotron3_<name> | <name> (so the existing
    `nemotron3_v030_resident_smoke.py` works as `touch ... v030_resident_smoke`)."""
    candidates = [f"nm3_{trigger_name}", f"nemotron3_{trigger_name}",
                  trigger_name]
    for cand in candidates:
        try:
            if cand in sys.modules:
                return importlib.reload(sys.modules[cand])
            return importlib.import_module(cand)
        except ImportError:
            continue
    return None


TRIG_DIR = PROJECT_ROOT / ".cache" / "nm3_runtime" / "trig"
LOG_PATH = TRIG_DIR / "last.log"
HARNESS_LOG_PATH = PROJECT_ROOT / ".cache" / "nm3_runtime" / "harness.log"
HARNESS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
_HARNESS_LOG_FH = open(HARNESS_LOG_PATH, "a", buffering=1)
_LAST_LOG_TS = time.time()


def log(msg: str):
    global _LAST_LOG_TS
    print(msg, flush=True)
    try:
        _HARNESS_LOG_FH.write(msg + "\n")
        _HARNESS_LOG_FH.flush()
    except Exception:
        pass
    try:
        with open(LOG_PATH, "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass
    _LAST_LOG_TS = time.time()


def _add_test_paths():
    for sub in TEST_SEARCH_DIRS:
        p = str(PROJECT_ROOT / sub)
        if p not in sys.path:
            sys.path.insert(0, p)


def run_test(state, trigger_name: str) -> int:
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    log(f"[harness] running {trigger_name}")
    try:
        # Reload server_nemotron3_nano_ttnn so code edits land without
        # re-bootstrap. Long-lived `state` keeps the weights resident.
        importlib.reload(base)
        mod = _discover_test_module(trigger_name)
        if mod is None:
            log(f"  ✗ {trigger_name}: no matching test module in {TEST_SEARCH_DIRS}")
            return 1
        if not hasattr(mod, "main"):
            log(f"  ✗ {trigger_name}: module {mod.__name__} has no main() entry point")
            return 1
        try:
            rc = mod.main(state=state)
        except TypeError:
            log("  ! main() doesn't accept state=; calling without (slow — will re-bootstrap)")
            rc = mod.main()
        if rc is None:
            rc = 0
        log(f"[harness] {trigger_name} exited rc={rc}")
        return int(rc)
    except Exception:
        log(traceback.format_exc())
        return 1


def main():
    _add_test_paths()
    TRIG_DIR.mkdir(parents=True, exist_ok=True)
    log(f"[harness] {time.strftime('%Y-%m-%d %H:%M:%S')} starting; "
        f"PROJECT_ROOT={PROJECT_ROOT}")
    log(f"[harness] env: UPLOAD_LAYERS={os.environ.get('NEMOTRON3_UPLOAD_LAYERS')}"
        f" MOE_MODE={os.environ.get('NEMOTRON3_MOE_MODE')}")
    log("[harness] bootstrapping Nemotron-3 Nano base state (~108s for all-resident)…")
    state = base.State()
    base.bootstrap(state, log)
    log(f"[harness] ready. Drop trigger files into {TRIG_DIR}/")
    log(f"[harness] examples: touch {TRIG_DIR}/v030_resident_smoke")
    log("[harness] special: _reload (re-import base), _exit (shutdown)")

    SKIP = {"last.log", "_reload", "_exit"}
    START_TS = time.time()
    while True:
        try:
            for trig in TRIG_DIR.iterdir():
                name = trig.name
                if name in SKIP or trig.is_dir():
                    continue
                try:
                    trig.unlink()
                except FileNotFoundError:
                    continue
                run_test(state, name)
            reload_trig = TRIG_DIR / "_reload"
            if reload_trig.exists():
                reload_trig.unlink(missing_ok=True)
                try:
                    importlib.reload(base)
                    log("[harness] _reload: server_nemotron3_nano_ttnn reloaded")
                except Exception:
                    log(traceback.format_exc())
            if (TRIG_DIR / "_exit").exists():
                log("[harness] _exit received; closing mesh and exiting")
                try:
                    import ttnn
                    ttnn.close_mesh_device(state.mesh)
                except Exception:
                    pass
                break
            if time.time() - _LAST_LOG_TS > 30:
                try:
                    qlen = len(list(TRIG_DIR.iterdir()))
                except Exception:
                    qlen = -1
                log(f"[harness] tick t={int(time.time() - START_TS)} queue={qlen}")
            time.sleep(0.5)
        except Exception:
            log(traceback.format_exc())
            time.sleep(0.5)


if __name__ == "__main__":
    sys.exit(main() or 0)
