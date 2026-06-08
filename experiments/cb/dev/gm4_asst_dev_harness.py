"""Interactive dev harness for the Gemma 4 12B IT spec-dec drafter
(`google/gemma-4-12b-it-assistant`) — bootstrap once, run many tests.

The drafter bootstrap is ~28s cold / ~10s warm (v0.1 measurement).
Without this harness every smoke pays the full cost; with it,
iterations drop to ~5-10s for the test itself.

Forks `nm3_dev_harness.py` verbatim with:
  - base = server_gemma4_12b_assistant_ttnn
  - test-name candidates: gm4_asst_<name> | gemma4_assistant_<name> | <name>
    so the existing `gemma4_assistant_forward_smoke.py` works as
    `touch ... forward_smoke`.
  - runtime dir: `gm4_asst_runtime`

REUSE: forks `nm3_dev_harness.py` per `[[feedback-reuse-mandate]]`.

Usage (per `[[reference-gm4-dev-harness]]` for Gemma 4):
  # one-time on qb1 (eats ~28s bootstrap, then idles):
  bash scripts/run_harness_tmux.sh gm4_asst qb1

  # per iteration (locally):
  bash scripts/deploy.sh experiments/serve/server_gemma4_12b_assistant_ttnn.py \\
                        experiments/cb/isolate/gemma4_assistant_forward_smoke.py
  ssh qb1 'touch ~/tt-xla/.cache/gm4_asst_runtime/trig/forward_smoke'
  ssh qb1 'cat ~/tt-xla/.cache/gm4_asst_runtime/trig/last.log'

The harness importlib.reloads server_gemma4_12b_assistant_ttnn on each
trigger, so server-side code changes land WITHOUT re-bootstrap.

Note: qb2's tt-metal build has the layernorm trisc1 SFPI regression
([[qb2-layernorm-trisc1-broken-2026-06-07]]); use qb1 for now.
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

import server_gemma4_12b_assistant_ttnn as base  # noqa: E402

TEST_SEARCH_DIRS = ["experiments/cb/validate", "experiments/cb/isolate",
                    "experiments/cb/bench", "experiments/cb/dev"]


def _discover_test_module(trigger_name: str):
    """Return the (re)loaded module for a trigger name, or None.
    Candidates: gm4_asst_<name> | gemma4_assistant_<name> | <name>
    (so existing `gemma4_assistant_forward_smoke.py` works as
    `touch ... forward_smoke`)."""
    candidates = [f"gm4_asst_{trigger_name}",
                  f"gemma4_assistant_{trigger_name}",
                  trigger_name]
    for cand in candidates:
        try:
            if cand in sys.modules:
                return importlib.reload(sys.modules[cand])
            return importlib.import_module(cand)
        except ImportError:
            continue
    return None


TRIG_DIR = PROJECT_ROOT / ".cache" / "gm4_asst_runtime" / "trig"
LOG_PATH = TRIG_DIR / "last.log"
HARNESS_LOG_PATH = PROJECT_ROOT / ".cache" / "gm4_asst_runtime" / "harness.log"
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


class _Tee:
    """Write to multiple file-likes so the smoke's prints land in
    trig/last.log AND the tmux session simultaneously."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
                st.flush()
            except Exception:
                pass
        return len(s)
    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


def run_test(state, trigger_name: str) -> int:
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    log(f"[harness] running {trigger_name}")
    last_fh = None
    saved_stdout = sys.stdout
    try:
        last_fh = open(LOG_PATH, "a", buffering=1)
        sys.stdout = _Tee(saved_stdout, last_fh)
        # Reload server_gemma4_12b_assistant_ttnn so code edits land
        # without re-bootstrap. Long-lived `state` keeps weights resident.
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
    finally:
        sys.stdout = saved_stdout
        if last_fh is not None:
            try:
                last_fh.close()
            except Exception:
                pass


def main():
    _add_test_paths()
    TRIG_DIR.mkdir(parents=True, exist_ok=True)
    log(f"[harness] {time.strftime('%Y-%m-%d %H:%M:%S')} starting; "
        f"PROJECT_ROOT={PROJECT_ROOT}")
    log("[harness] bootstrapping drafter (gemma-4-12b-it-assistant ~380M) "
        "— v0.1 measured ~28s cold / ~10s warm…")
    state = base.State()
    base.bootstrap(state, log)
    log(f"[harness] ready. Drop trigger files into {TRIG_DIR}/")
    log(f"[harness] examples: touch {TRIG_DIR}/forward_smoke")
    log("[harness]           touch ~/tt-xla/.cache/gm4_asst_runtime/trig/embed_smoke")
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
                    log("[harness] _reload: server_gemma4_12b_assistant_ttnn reloaded")
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
