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
CB_DIR = REPO / "experiments" / "cb"
KNOWN_DIRS = [SERVE_DIR, REPO / "experiments" / "utils"]
# Every entrypoint a daemon or `make dr` user might launch. Each one's
# transitive bare imports must land in deploy.sh.
ENTRY_GLOBS = [
    SERVE_DIR.glob("server*.py"),
    SERVE_DIR.glob("cb_*.py"),
    SERVE_DIR.glob("openai_endpoint.py"),
    *(CB_DIR.glob(pat) for pat in ("validate/*.py", "bench/*.py", "load/*.py", "needle.py")),
]


def main() -> int:
    internal: dict[str, Path] = {}
    for d in KNOWN_DIRS:
        for p in d.glob("*.py"):
            internal[p.stem] = p.relative_to(REPO)

    # Read deploy.sh default arg list (regex matches paths globbed OR explicit).
    deploy_text = (REPO / "scripts" / "deploy.sh").read_text()
    explicit_paths = set(re.findall(r"experiments/\S+\.py", deploy_text))
    glob_patterns = re.findall(r"experiments/\S+/\*\.py", deploy_text)

    def covered(rel_path: str) -> bool:
        if rel_path in explicit_paths:
            return True
        # A glob `experiments/serve/*.py` covers any direct child .py file.
        for g in glob_patterns:
            prefix = g[: -len("/*.py")]
            p = Path(rel_path)
            if p.parent == Path(prefix):
                return True
        return False

    needed: set[str] = set()
    seen_entries: list[Path] = []
    for glob in ENTRY_GLOBS:
        for src in glob:
            seen_entries.append(src)
            for node in ast.walk(ast.parse(src.read_text())):
                if (isinstance(node, ast.ImportFrom)
                        and node.module is not None
                        and node.level == 0
                        and "." not in node.module
                        and node.module in internal):
                    needed.add(str(internal[node.module]))
            # The entry file itself also needs to be deployed.
            needed.add(str(src.relative_to(REPO)))

    missing = sorted(p for p in needed if not covered(p))
    if missing:
        print("FAIL: scripts/deploy.sh default args do NOT cover these paths "
              "(entry files or their internal imports):")
        for m in missing:
            print(f"  - {m}")
        print("\nFix: add the path(s) above to the `set --` block in scripts/deploy.sh, "
              "or extend a glob to cover them.")
        return 1
    print(f"OK: deploy.sh covers all {len(needed)} entry files + internal imports "
          f"(across {len(seen_entries)} entrypoints).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
