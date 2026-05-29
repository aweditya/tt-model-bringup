"""JAX plugin registration for Tenstorrent Blackhole.

JAX discovers plugins via the jax_plugins namespace package entry point.
When JAX imports this module, it calls initialize() which registers our
PJRT plugin shared library.

Usage:
  # Just importing jax after installing the plugin is enough:
  import jax
  print(jax.devices())  # Should show TtDevice(id=0)

  # Or register manually:
  from jax_plugins.tt import initialize
  initialize()
"""

import os
import jax._src.xla_bridge as xb


def initialize():
    """Register the TT PJRT plugin with JAX."""
    # Look for the shared library relative to this file.
    # After building with CMake, the .so is in the build directory.
    # For development, we also check common locations.
    plugin_dir = os.path.dirname(__file__)
    candidates = [
        os.path.join(plugin_dir, "libpjrt_plugin_tt.so"),
        os.path.join(plugin_dir, "..", "..", "build", "libpjrt_plugin_tt.so"),
        os.path.join(
            plugin_dir, "..", "..", "build", "lib", "libpjrt_plugin_tt.so"
        ),
    ]

    library_path = None
    for path in candidates:
        if os.path.exists(path):
            library_path = os.path.abspath(path)
            break

    if library_path is None:
        raise RuntimeError(
            f"Could not find libpjrt_plugin_tt.so. Searched:\n"
            + "\n".join(f"  {c}" for c in candidates)
            + "\nBuild the plugin first: cd pjrt_plugin && cmake -B build && cmake --build build"
        )

    # priority=500 means JAX prefers this over CPU (priority=0) but not
    # over CUDA (priority=300) if CUDA is available. Adjust as needed.
    xb.register_plugin("tt", priority=500, library_path=library_path, options=None)
