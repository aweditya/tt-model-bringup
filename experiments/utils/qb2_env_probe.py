"""Print qb2 (and qb1) environment fingerprint relevant to the SFPI/layernorm
regression investigation 2026-06-08.

Reports:
- python version
- transformers / torch / ttnn versions
- TT_METAL_HOME + TT_BUILD_DIR resolved env
- SFPI version file content (best-effort)
- pre-compiled FW hash dirs present

No device required. Intended to be run via `scripts/run_remote_qb2.sh` (qb2)
or directly under any shell as `.venv/bin/python experiments/utils/qb2_env_probe.py`.
"""
from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path


def _line(k: str, v: str) -> None:
    print(f"{k:32s} {v}")


def main() -> int:
    _line("python", sys.version.split()[0])
    for mod in ("transformers", "torch"):
        try:
            m = __import__(mod)
            _line(f"{mod}", getattr(m, "__version__", "?"))
        except Exception as exc:
            _line(f"{mod}", f"IMPORT FAIL: {exc}")
    # ttnn is optional in case env vars aren't set; do not blow up.
    try:
        import ttnn  # type: ignore[import-not-found]
        _line("ttnn", getattr(ttnn, "__version__", "?"))
    except Exception as exc:
        _line("ttnn", f"IMPORT FAIL: {exc}")

    for ev in ("TT_METAL_HOME", "TT_BUILD_DIR", "ARCH_NAME"):
        _line(f"env:{ev}", os.environ.get(ev, "<unset>"))

    tt_home = Path(os.environ.get("TT_METAL_HOME", str(Path.home() / "tenstorrent/tt-metal")))
    vfile = tt_home / "tt_metal" / "sfpi-version"
    if vfile.is_file():
        text = vfile.read_text().strip().splitlines()
        head = next((l for l in text if l.startswith("# sfpi")), "?")
        ver_line = next((l for l in text if l.startswith("sfpi_version=")), "?")
        bld_line = next((l for l in text if l.startswith("sfpi_build=")), "?")
        _line("sfpi.release", head)
        _line("sfpi.version", ver_line)
        _line("sfpi.build", bld_line)
    else:
        _line("sfpi-version", f"missing at {vfile}")

    sfpi_bin = tt_home / "runtime/sfpi/compiler/bin/riscv-tt-elf-g++"
    if sfpi_bin.is_file():
        try:
            r = subprocess.run(["md5sum", str(sfpi_bin)], capture_output=True, text=True, timeout=10)
            _line("sfpi.gpp.md5", r.stdout.split()[0] if r.returncode == 0 else r.stderr.strip())
        except Exception as exc:
            _line("sfpi.gpp.md5", f"err: {exc}")
        st = sfpi_bin.stat()
        _line("sfpi.gpp.mtime", str(st.st_mtime_ns))

    precomp_root = tt_home / "tt_metal/pre-compiled"
    if precomp_root.is_dir():
        hashes = sorted(p.name for p in precomp_root.iterdir() if p.is_dir())
        _line("precompiled.count", str(len(hashes)))
        if hashes:
            _line("precompiled.first", hashes[0])
            _line("precompiled.last", hashes[-1])

    # tt-smi version
    try:
        r = subprocess.run(["tt-smi", "-v"], capture_output=True, text=True, timeout=10)
        _line("tt-smi", (r.stdout or r.stderr).splitlines()[0] if r.stdout or r.stderr else "?")
    except Exception as exc:
        _line("tt-smi", f"err: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
