"""Interactive dev harness for CB35 — bootstrap 35B once, run many tests.

The 35B weight upload is ~14 min. Without this harness every fix-test
cycle pays that cost. Here we bootstrap once into a long-lived python
process and run tests on demand via trigger files. Iteration becomes:

  EDIT (local) → deploy.sh → ssh qb1 'touch tt-xla/.cache/cb35_runtime/trig/<test_name>'
  → read tt-xla/.cache/cb35_runtime/trig/last.log

Each trigger:
  1. `importlib.reload`s `server_35b_cb` and the test module so the new
     code is picked up without re-bootstrap.
  2. Calls `test_module.main(state=state)` — tests must accept a
     pre-bootstrapped state.
  3. Catches all exceptions; harness survives bad tests.

Usage:
  # one-time on qb1 (eats 14 min bootstrap, then idles):
  bash scripts/run_harness_tmux.sh

  # per iteration (locally):
  bash scripts/deploy.sh experiments/serve/server_35b_cb.py experiments/cb/validate/cb35_v0_smoke.py
  ssh qb1 'touch tt-xla/.cache/cb35_runtime/trig/v0_smoke'
  ssh qb1 'cat tt-xla/.cache/cb35_runtime/trig/last.log'
"""
from __future__ import annotations

import importlib
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "experiments" / "serve").is_dir())
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import server_35b_ttnn as base  # noqa: E402

# Tests are discovered dynamically by trigger-file basename. For trigger
# 'foo', the harness searches for a module named (in order) cb35_foo,
# cb35_v0_foo, foo across the test directories. Drop a file with
# `main(state=None)` + touch the trigger; no harness restart needed.
TEST_SEARCH_DIRS = ["experiments/cb/validate", "experiments/cb/isolate",
                    "experiments/cb/bench", "experiments/cb/dev"]


def _discover_test_module(trigger_name: str):
    """Return the (re)loaded module for a trigger name, or None."""
    candidates = [f"cb35_{trigger_name}", f"cb35_v0_{trigger_name}", trigger_name]
    for cand in candidates:
        try:
            if cand in sys.modules:
                return importlib.reload(sys.modules[cand])
            return importlib.import_module(cand)
        except ImportError:
            continue
    return None

TRIG_DIR = PROJECT_ROOT / ".cache" / "cb35_runtime" / "trig"
LOG_PATH = TRIG_DIR / "last.log"
HARNESS_LOG_PATH = PROJECT_ROOT / ".cache" / "cb35_runtime" / "harness.log"
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
    """Add test directories to sys.path so importlib.import_module finds them."""
    for sub in TEST_SEARCH_DIRS:
        p = str(PROJECT_ROOT / sub)
        if p not in sys.path:
            sys.path.insert(0, p)


def run_test(state, trigger_name: str) -> int:
    """Discover + reload + run a test for the given trigger name."""
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    log(f"[harness] running {trigger_name}")
    try:
        # Reload server_35b_cb — it's the code under test for all CB35 tests.
        import server_35b_cb  # noqa: F401
        importlib.reload(server_35b_cb)
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
    log("[harness] bootstrapping 35B base state (~14 min)…")
    state = base.State()
    base.bootstrap(state, log)
    log(f"[harness] ready. Drop trigger files into {TRIG_DIR}/")
    log(f"[harness] examples: touch {TRIG_DIR}/v0_smoke  or  touch {TRIG_DIR}/v0_chat")
    log("[harness] special: _reload (re-import server_35b_cb), _exit (shutdown)")

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
            # Special trigger: 'reload' = importlib-reload server_35b_cb without running anything
            reload_trig = TRIG_DIR / "_reload"
            if reload_trig.exists():
                reload_trig.unlink(missing_ok=True)
                try:
                    import server_35b_cb
                    importlib.reload(server_35b_cb)
                    log("[harness] _reload: server_35b_cb reloaded")
                except Exception:
                    log(traceback.format_exc())
            # Special trigger: 'exit' = clean shutdown
            if (TRIG_DIR / "_exit").exists():
                log("[harness] _exit received; closing mesh and exiting")
                try:
                    import ttnn
                    ttnn.close_device(state.mesh)
                except Exception:
                    pass
                break
            # Heartbeat: if no log line in >30s, emit one so hangs are visible.
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
