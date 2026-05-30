"""CI guard: every internal module that `experiments/serve/server*.py` imports
must be in `scripts/deploy.sh`'s default arg set.

Catches the clone-and-run audit's P0 #1 failure mode (audit doc, 2026-05-30):
`server_tp.py` does `from full_layer_tp_probe import ...` (12 sites) — that
module lives in `experiments/utils/`. If `deploy.sh` doesn't list it, `make dr`
on a fresh box rsyncs server_tp without its sibling probe → server crashes
mid-bootstrap with ModuleNotFoundError, which a user reads as garbage output.

Pure static — uses `ast` + regex, no imports of ttnn or any device-tied module.
Suitable for a CI smoke step in vanilla Python.

    python scripts/ci_check_deploy_sync.py
    → exits 0 if every internal import resolves; 1 + a diff if any are missing.
"""
import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SERVE_DIR = REPO / "experiments" / "serve"
KNOWN_DIRS = [SERVE_DIR, REPO / "experiments" / "utils"]


def main() -> int:
    # Build map of `internal_module_name -> repo-relative path` from the dirs
    # server*.py code can resolve via its bootstrap-time sys.path inserts.
    internal: dict[str, Path] = {}
    for d in KNOWN_DIRS:
        for p in d.glob("*.py"):
            internal[p.stem] = p.relative_to(REPO)

    # Read deploy.sh default arg list (the `if (( $# == 0 )); then set -- ... fi`).
    deploy_text = (REPO / "scripts" / "deploy.sh").read_text()
    default_args = set(re.findall(r"experiments/\S+\.py", deploy_text))

    # Walk every server*.py; collect bare `from X import ...` where X is an
    # internal-module name (resolvable via the sys.path inserts in bootstrap).
    needed: set[str] = set()
    for src in sorted(SERVE_DIR.glob("server*.py")):
        text = src.read_text()
        for node in ast.walk(ast.parse(text)):
            if (isinstance(node, ast.ImportFrom)
                    and node.module is not None
                    and node.level == 0
                    and "." not in node.module
                    and node.module in internal):
                needed.add(str(internal[node.module]))

    missing = needed - default_args
    if missing:
        print("FAIL: scripts/deploy.sh default args do NOT sync these internal "
              "modules imported by experiments/serve/server*.py:")
        for m in sorted(missing):
            print(f"  - {m}")
        print("\nFix: add the path(s) above to the `set --` block in scripts/deploy.sh.")
        return 1
    print(f"OK: scripts/deploy.sh covers all {len(needed)} internal modules "
          f"imported by server*.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
