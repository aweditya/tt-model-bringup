"""Interactive dev harness for Gemma 4 12B — bootstrap once, run many tests.

Forks `experiments/cb/dev/cb35_dev_harness.py` (line-for-line where
possible; differences are: import `server_gemma4_unified_ttnn` instead
of `server_35b_ttnn`, no `server_35b_cb` reload — Gemma 4 doesn't have
a CB module yet, that's v1). Bootstrap is ~80 sec (vs 14 min for 35B),
so this matters less than for 35B but still saves ~80 sec per iteration.

Trigger names are mapped through these candidates per trigger:
  gm4_<trigger>, gm4_v033a_<trigger>, gm4_v032_<trigger>, gm4_v031_<trigger>, <trigger>
so e.g. `touch trig/long_cos` matches `gm4_v033a_long_cos.py`.

Usage:
  # one-time on qb1 (eats 80s bootstrap, then idles):
  bash scripts/run_harness_tmux.sh gm4   # see scripts/run_harness_tmux.sh

  # per iteration (locally):
  bash scripts/deploy.sh experiments/serve/server_gemma4_unified_ttnn.py \\
      experiments/cb/isolate/gm4_v033a_long_cos.py
  ssh qb1 'touch tt-xla/.cache/gm4_runtime/trig/v033a_long_cos'
  ssh qb1 'cat tt-xla/.cache/gm4_runtime/trig/last.log'

Special triggers:
  _reload  — importlib-reload `server_gemma4_unified_ttnn` (no test run)
  _exit    — clean shutdown (closes mesh, exits)
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

import server_gemma4_unified_ttnn as base  # noqa: E402

TEST_SEARCH_DIRS = ["experiments/cb/validate", "experiments/cb/isolate",
                    "experiments/cb/bench", "experiments/cb/dev"]


def _discover_test_module(trigger_name: str):
    """Return the (re)loaded module for a trigger name, or None."""
    candidates = [
        f"gm4_{trigger_name}",
        f"gm4_v033a_{trigger_name}",
        f"gm4_v033b_{trigger_name}",
        f"gm4_v033c_{trigger_name}",
        f"gm4_v032_{trigger_name}",
        f"gm4_v031_{trigger_name}",
        f"gm4_v030_{trigger_name}",
        trigger_name,
    ]
    for cand in candidates:
        try:
            if cand in sys.modules:
                return importlib.reload(sys.modules[cand])
            return importlib.import_module(cand)
        except ImportError:
            continue
    return None


TRIG_DIR = PROJECT_ROOT / ".cache" / "gm4_runtime" / "trig"
LOG_PATH = TRIG_DIR / "last.log"
HARNESS_LOG_PATH = PROJECT_ROOT / ".cache" / "gm4_runtime" / "harness.log"
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
        # Reload the server module so probes see any in-flight edits.
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
    log("[harness] bootstrapping Gemma 4 12B base state (~80 sec)…")
    state = base.State()
    base.bootstrap(state, log)
    log(f"[harness] ready. Drop trigger files into {TRIG_DIR}/")
    log(f"[harness] e.g.: touch {TRIG_DIR}/v031_multistep_cos")
    log(f"[harness] special: _reload, _exit")

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
                    log("[harness] _reload: server_gemma4_unified_ttnn reloaded")
                except Exception:
                    log(traceback.format_exc())
            if (TRIG_DIR / "_exit").exists():
                log("[harness] _exit received; closing mesh and exiting")
                try:
                    import ttnn
                    ttnn.close_device(state.mesh)
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
    main()
