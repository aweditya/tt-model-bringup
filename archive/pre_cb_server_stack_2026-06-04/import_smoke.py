"""Device-gated import smoke for the 27B server modules.

Validates the package-import wiring after the 91f/91l -> ondevice_27b/generate_27b
untangle: both modules must import via `from experiments.serve import ...` (the
same path the servers use under `-m experiments.serve.server_tp`) and expose the
symbols the servers dereference. Imports ttnn but opens no mesh device, so it is
safe and fast. Run on a TT host from the repo root:

    scripts/run_remote.sh --no-reset -m experiments.serve.import_smoke
"""
from experiments.serve import generate_27b, ondevice_27b

ONDEVICE_SYMS = [
    "upload", "load_layer_weights_all", "hifi4",
    "mlp_step_ondevice", "deltanet_step_ondevice",
    "gated_attn_step_ondevice", "gated_attn_step_ondevice_paged",
]
GENERATE_SYMS = ["load_embed_lm_head_weights"]

missing = [f"ondevice_27b.{s}" for s in ONDEVICE_SYMS if not hasattr(ondevice_27b, s)]
missing += [f"generate_27b.{s}" for s in GENERATE_SYMS if not hasattr(generate_27b, s)]
if missing:
    raise SystemExit(f"IMPORT SMOKE FAIL - missing symbols: {missing}")
print("IMPORT SMOKE OK - ondevice_27b + generate_27b import via experiments.serve "
      "and expose every symbol the servers use")
