# Wiki 53: Official TT PJRT Plugin — Installs But Crashes

## What We Tested

```bash
pip install pjrt-plugin-tt --extra-index-url https://pypi.eng.aws.tenstorrent.com/
```

Version 0.3.0 installed (187 MB wheel). Contains bundled `libtt_metal.so` and `libTTMLIRRuntime.so`.

## What Happened

```python
import jax
print(jax.devices())
```

Output:
```
WARNING: Platform 'tt' is experimental and not all JAX functionality may be correctly supported!
# ... opens device driver, detects both Blackhole chips ...
Signal: Segmentation fault (11)
```

The crash is in `convert_1d_mesh_adjacency_to_row_major_vector` during `SystemMesh::create()`. The plugin's bundled tt-metal tries to set up a multi-device mesh topology and fails.

## Root Cause (Likely)

Version mismatch between the plugin's bundled tt-metal libraries and the host's firmware/driver:
- Host firmware: 19.6.0
- Host tt-metal (ttnn): 0.68.0
- Plugin's bundled tt-metal: unknown (likely different version)

The plugin bundles its own `libtt_metal.so` which may expect a different firmware version or mesh configuration.

## What This Tells Us

1. **The plugin is real and actively developed** — it installs, registers with JAX, and attempts to open the device
2. **Blackhole is recognized** — the driver loads, finds both chips, maps hugepages
3. **The crash is in mesh topology setup** — not in basic device access
4. **Fix likely requires matching tt-metal versions** — install the plugin version that matches our firmware, or update firmware to match the plugin

## Next Steps

- [ ] Check TT's release notes for which firmware versions are compatible with pjrt-plugin-tt 0.3.0
- [ ] Try older plugin versions (`pip install pjrt-plugin-tt==0.2.0` etc.)
- [ ] File a bug report with TT including the crash trace
- [ ] Try setting `TT_METAL_DEVICE_IDS=0` to use only one chip (avoids mesh topology)
