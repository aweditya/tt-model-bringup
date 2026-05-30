# ttnn JIT Compilation Bug — bf16 / fp32 Weight Path

**Date**: 2026-05-12
**Status**: Known limitation; workaround in place.

## Symptom

When `experiments/91p_ttnn_layer0_vs_hf.py --weight-dtype bf16` or `fp32` is run,
ttnn JIT compilation of the `layernorm_large_tensor` kernel (and several
SFPU kernels) fails with:

```
trisc1 build failed.
ckernel_sfpu_rsqrt_compat.h:119:84: error: cannot convert 'int' to 'sfpi::RoundMode' [-Wtemplate-body]
ckernel_sfpu_exp.h:88:63: error: cannot convert 'int' to 'sfpi::RoundMode' [-Wtemplate-body]
ckernel_sfpu_log.h:56:50: error: cannot convert 'int' to 'sfpi::RoundMode'
...
```

The error is in ttnn's low-level kernel library (LLK) headers, not in our
code. It triggers when the JIT must compile a fresh kernel variant for a
weight dtype that wasn't pre-built in the wheel.

## What triggers it

- `--weight-dtype bf16` triggers compilation of layernorm + matmul kernels for
  bf16-weight × fp32-activation × fp32-output. Not pre-built → compile fails.
- `--weight-dtype fp32` triggers fp32-weight × fp32-activation. Also fails.
- `--weight-dtype bf8` (default) uses kernels that are already in the
  ttnn pre-compiled cache. **Works.**

`JIT cache stats: 0/17 hits` confirms ttnn has to build new kernels for the
non-bf8 path.

## Implications

We cannot easily ablate weight precision to test whether bf8 quantization is
the source of the residual 0.3% drift between our layer 0 output and HF's
reference. The ablation hypothesis is blocked behind a ttnn LLK header bug.

## Workarounds (none applied yet)

1. **Bisect the ttnn LLK headers** — the bug is in template instantiation
   that converts `int` to `sfpi::RoundMode`. May be fixable as a small upstream patch.
2. **Wait for a ttnn release** that pre-compiles non-bf8 weight kernels.
3. **Find an alternative kernel path** — maybe ttnn has a "small" variant of
   rms_norm that doesn't hit this code. Less likely.
4. **Switch validation strategy** — instead of changing weight dtype, do
   substep-level diff: HF substep activations vs ttnn substep activations,
   localize the most-divergent step. This is what we'll do.

## Reproduction

```bash
ssh qb2
cd ~/tt-xla
HF_HOME=$HOME/tt-xla/.cache/hf .venv/bin/python \
    experiments/91p_ttnn_layer0_vs_hf.py --weight-dtype bf16
# crashes during JIT build of layernorm_large_tensor
```

## Footnote for the wiki

Add to `bringup_checklist.md` (planned): when ablating weight precision, expect
ttnn's JIT to need to compile new kernels for any combination not in the
default cache. Verify those kernels compile BEFORE running expensive
experiments.
