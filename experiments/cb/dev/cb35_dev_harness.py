"""Interactive dev harness for CB35 — bootstrap 35B once, run many tests.

The 35B weight upload is ~14 min. Without this harness every fix-test
cycle pays that cost. Here we bootstrap once into a long-lived python
process and run tests on demand via trigger files. Iteration becomes:

  EDIT (local) → deploy.sh → ssh qb1 'touch /tmp/cb35_trig/<test_name>'
  → read /tmp/cb35_trig/last.log

Each trigger:
  1. `importlib.reload`s `server_35b_cb` and the test module so the new
     code is picked up without re-bootstrap.
  2. Calls `test_module.main(state=state)` — tests must accept a
     pre-bootstrapped state.
  3. Catches all exceptions; harness survives bad tests.

Usage:
  # one-time on qb1 (eats 14 min bootstrap, then idles):
  bash scripts/run_remote.sh --no-reset experiments/cb/dev/cb35_dev_harness.py

  # per iteration (locally):
  bash scripts/deploy.sh experiments/serve/server_35b_cb.py experiments/cb/validate/cb35_v0_smoke.py
  ssh qb1 'mkdir -p /tmp/cb35_trig && touch /tmp/cb35_trig/v0_smoke'
  ssh qb1 'cat /tmp/cb35_trig/last.log'

Add new tests by registering them in TESTS below.
"""
from __future__ import annotations

import importlib
import os
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "experiments" / "serve").is_dir())
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import server_35b_ttnn as base  # noqa: E402

# Tests registered by trigger-file basename. Each value is the importable
# module path; the module must expose `main(state=None)`. We import lazily
# inside the trigger loop so editing a test module + rsyncing it picks up
# the new code via importlib.reload.
TESTS: dict[str, str] = {
    "v0_smoke": "cb35_v0_smoke",  # experiments/cb/validate/cb35_v0_smoke.py
}

TRIG_DIR = Path("/tmp/cb35_trig")
LOG_PATH = TRIG_DIR / "last.log"


def log(msg: str):
    print(msg, flush=True)
    try:
        with open(LOG_PATH, "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def _add_test_paths():
    """Add test directories to sys.path so importlib.import_module finds them."""
    for sub in ("experiments/cb/validate", "experiments/cb/isolate"):
        p = str(PROJECT_ROOT / sub)
        if p not in sys.path:
            sys.path.insert(0, p)


def run_test(state, name: str, module_path: str) -> int:
    """Reload module + run main(state=state). Returns exit code (0 = pass)."""
    # Clear log
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    log(f"[harness] running {name} (module={module_path})")
    try:
        # Reload server_35b_cb first — it's the code under test for all CB35 tests.
        import server_35b_cb  # noqa: F401
        importlib.reload(server_35b_cb)
        # Reload the test module itself.
        if module_path in sys.modules:
            mod = importlib.reload(sys.modules[module_path])
        else:
            mod = importlib.import_module(module_path)
        if not hasattr(mod, "main"):
            log(f"  ✗ {name}: module has no main() entry point")
            return 1
        # Tests must accept state= kwarg.
        try:
            rc = mod.main(state=state)
        except TypeError:
            log("  ! test module's main() doesn't accept state=; calling without it "
                "(will likely re-bootstrap — slow)")
            rc = mod.main()
        if rc is None:
            rc = 0
        log(f"[harness] {name} exited rc={rc}")
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
    log(f"[harness] ready. Available tests: {sorted(TESTS)}")
    log(f"[harness] trigger via: touch {TRIG_DIR}/<test_name>")

    while True:
        for name, module_path in TESTS.items():
            trig = TRIG_DIR / name
            if trig.exists():
                try:
                    trig.unlink()
                except FileNotFoundError:
                    pass
                run_test(state, name, module_path)
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
        time.sleep(0.5)


if __name__ == "__main__":
    sys.exit(main() or 0)
