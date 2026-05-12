#!/usr/bin/env python3
"""
Probe ttnn for Tracy profiler support.

Tracy gives per-op kernel timing on the device, separated from host
dispatch overhead — exactly what sync-bounded host timing CAN'T measure.

This probe answers:
  1. Does the installed ttnn package have Tracy support compiled in?
  2. Are there Python-side Tracy hooks we can use (ttnn.tracy_*, etc.)?
  3. Are there environment variables that enable/disable Tracy?
  4. What output format does the ttnn build produce (Tracy file format)?

Run on qb2 — pure introspection, no device required:
    cd ~/tt-xla && .venv/bin/python experiments/utils/tracy_availability_probe.py
"""
import os, sys, importlib, inspect


def probe_ttnn_attrs():
    """List ttnn module attributes that mention 'tracy', 'profile', 'trace'."""
    import ttnn
    print(f"\nttnn package location: {ttnn.__file__}")
    print(f"ttnn dir() — entries matching tracy/profile/timeline:")
    matches = sorted(a for a in dir(ttnn)
                     if any(x in a.lower() for x in ['tracy', 'profile', 'timeline', 'perf']))
    for a in matches:
        try:
            val = getattr(ttnn, a)
            kind = type(val).__name__
            print(f"  ttnn.{a:<40s}  ({kind})")
        except Exception as e:
            print(f"  ttnn.{a:<40s}  (err: {e})")
    return matches


def probe_tt_metal():
    """Some Tracy hooks live in tt_metal directly."""
    try:
        import ttnn._ttnn  # the C++ binding module
        print(f"\nttnn._ttnn loaded")
        matches = sorted(a for a in dir(ttnn._ttnn)
                         if any(x in a.lower() for x in ['tracy', 'profile', 'perf']))
        for a in matches:
            print(f"  ttnn._ttnn.{a}")
    except ImportError as e:
        print(f"\nttnn._ttnn import failed: {e}")


def probe_env():
    """Tracy is often controlled by env vars."""
    candidates = [
        "TT_METAL_ENABLE_TRACY", "TRACY_ENABLE", "TT_METAL_DEVICE_PROFILER",
        "TT_METAL_DPRINT_RISCV", "TT_METAL_DPRINT_LAYER_GAP_CYCLES",
        "TT_METAL_LOGGER_LEVEL", "TRACY_NO_INVARIANT_CHECK",
    ]
    print("\nRelevant environment variables (set / unset):")
    for var in candidates:
        val = os.environ.get(var, "<unset>")
        print(f"  {var:<40s} = {val}")


def probe_shared_libs():
    """Tracy is sometimes statically linked into libtt_metal.so."""
    import ttnn
    ttnn_dir = os.path.dirname(ttnn.__file__)
    print(f"\nScanning {ttnn_dir} for shared libs matching tracy/profiler…")
    found = []
    for root, _, files in os.walk(ttnn_dir):
        for f in files:
            if f.endswith(('.so', '.dylib')) and any(
                x in f.lower() for x in ['tracy', 'profiler']
            ):
                found.append(os.path.join(root, f))
    if found:
        for f in found:
            print(f"  {f}")
    else:
        print("  (no shared libs explicitly named tracy/profiler)")
    return found


def probe_build_info():
    """ttnn's build log/info often mentions which features are enabled."""
    try:
        import ttnn
        # Some ttnn versions expose a build info function
        for name in ("get_build_info", "build_info", "_build_info"):
            if hasattr(ttnn, name):
                fn = getattr(ttnn, name)
                print(f"\nttnn.{name}:")
                try:
                    print(f"  {fn() if callable(fn) else fn}")
                except Exception as e:
                    print(f"  (call failed: {e})")
    except Exception as e:
        print(f"\nbuild info probe failed: {e}")


def probe_device_profiler():
    """tt_metal Device API may have a profiler interface."""
    try:
        from ttnn import (
            device as _dev_module,
        )
        for name in dir(_dev_module):
            if any(x in name.lower() for x in ['tracy', 'profile', 'perf']):
                print(f"  ttnn.device.{name}")
    except Exception:
        pass


def main():
    print("=" * 64)
    print("Probe: Tracy availability in the installed ttnn")
    print("=" * 64)

    probe_env()
    matches = probe_ttnn_attrs()
    probe_tt_metal()
    probe_shared_libs()
    probe_build_info()
    probe_device_profiler()

    print("\n" + "=" * 64)
    print("Verdict:")
    if matches:
        print("  ttnn has SOME profiling/Tracy-named entry points — see attrs above.")
        print("  Next step: pick the most likely candidate and probe its signature.")
    else:
        print("  No obvious Tracy entry points in ttnn Python API.")
        print("  Tracy may still be in the build (controlled via env var). Try:")
        print("    TT_METAL_DEVICE_PROFILER=1 python <your_workload>")
        print("  and look for tracy file output in cwd or tmp.")


if __name__ == "__main__":
    main()
