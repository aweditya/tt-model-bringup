#!/usr/bin/env python3
"""Patch tracy/process_ops_logs.py so the 'Device data missing' assertion at
line ~561 becomes a skip. Lets us get a partial merged CSV when trace-replay
op IDs don't fully match host-side records (the failure mode we hit on the
35B decode).

Idempotent — safe to run multiple times.

Usage:
  python3 _patch_tracy_assertion.py [path/to/process_ops_logs.py]
"""
import sys
from pathlib import Path

DEFAULT = "/home/aditya/tt-xla/.venv/lib/python3.10/site-packages/tracy/process_ops_logs.py"

OLD = '''            assert candidates, (
                f"Device data missing: Op {op_id} not present in {PROFILER_CPP_DEVICE_PERF_REPORT} "
                f"for device {device_id} (trace_id={host_trace_id})"
            )
'''

NEW = '''            if not candidates:
                # Patched 2026-05-25 — skip ops with no matching device perf
                # data instead of asserting. Lets us get a partial merged CSV
                # for traced runs where trace-replay op IDs don't always match
                # host-side host_op_ids.
                continue
'''


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
    src = path.read_text()
    marker = "Patched 2026-05-25 — skip ops with no matching device perf"
    if marker in src:
        print("Already patched (no-op).")
        return
    if OLD not in src:
        print("PATTERN NOT FOUND — file structure differs from expected. Aborting.")
        sys.exit(1)
    backup = path.with_suffix(path.suffix + ".pre_patch_bak")
    if not backup.exists():
        backup.write_text(src)
        print(f"Backup written to {backup}")
    path.write_text(src.replace(OLD, NEW))
    print(f"Patched {path}")


if __name__ == "__main__":
    main()
