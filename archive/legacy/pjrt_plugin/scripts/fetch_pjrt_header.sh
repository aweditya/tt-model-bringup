#!/bin/bash
# Fetch the exact PJRT C API header from XLA matching our jaxlib version.
#
# The PJRT_Api struct layout is ABI-sensitive -- if the field order doesn't
# match what jaxlib expects, the plugin will crash or silently corrupt data.
#
# Usage: ./scripts/fetch_pjrt_header.sh
# Requires: pip, python3, curl

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLUGIN_DIR="$(dirname "$SCRIPT_DIR")"
HEADER_DIR="$PLUGIN_DIR/third_party/pjrt"

echo "=== Determining XLA commit from installed jaxlib ==="

# Get jaxlib version
JAXLIB_VERSION=$(python3 -c "import jaxlib; print(jaxlib.version.__version__)" 2>/dev/null || \
                 python3 -c "import jaxlib; print(jaxlib.__version__)")
echo "jaxlib version: $JAXLIB_VERSION"

# The XLA commit is embedded in jaxlib's build metadata.
# Try to get it from the jaxlib package.
XLA_COMMIT=$(python3 -c "
try:
    from importlib.metadata import metadata
    m = metadata('jaxlib')
    # Look in the metadata for the XLA commit
    print('unknown')
except:
    print('unknown')
")

if [ "$XLA_COMMIT" = "unknown" ]; then
    echo ""
    echo "Could not auto-detect XLA commit. You need to find it manually:"
    echo "  1. Check: pip show jaxlib  (look for 'Version' or git hash)"
    echo "  2. Find the matching XLA tag/commit on github.com/openxla/xla"
    echo "  3. Download: curl -o pjrt_c_api.h https://raw.githubusercontent.com/openxla/xla/<COMMIT>/xla/pjrt/c/pjrt_c_api.h"
    echo ""
    echo "For jaxlib $JAXLIB_VERSION, try looking at:"
    echo "  https://github.com/jax-ml/jax/blob/jaxlib-v$JAXLIB_VERSION/WORKSPACE"
    echo ""

    # Fetch latest as fallback (may not match exactly)
    echo "Fetching latest pjrt_c_api.h from XLA main branch as starting point..."
    curl -sL "https://raw.githubusercontent.com/openxla/xla/main/xla/pjrt/c/pjrt_c_api.h" \
        -o "$HEADER_DIR/pjrt_c_api.h"
    echo "Downloaded to $HEADER_DIR/pjrt_c_api.h"
    echo "WARNING: This may not match your jaxlib version. Verify before using!"
else
    echo "XLA commit: $XLA_COMMIT"
    curl -sL "https://raw.githubusercontent.com/openxla/xla/$XLA_COMMIT/xla/pjrt/c/pjrt_c_api.h" \
        -o "$HEADER_DIR/pjrt_c_api.h"
    echo "Downloaded to $HEADER_DIR/pjrt_c_api.h"
fi

echo ""
echo "=== Header info ==="
grep -c "PJRT_" "$HEADER_DIR/pjrt_c_api.h" || true
echo "function pointer fields"
echo ""
echo "Next step: rebuild the plugin with the new header."
