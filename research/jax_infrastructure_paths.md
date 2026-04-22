# JAX Infrastructure: Three Paths Forward

## Current: Jaxpr Interpreter + Trace Capture
- **Status:** Working, 28 ops, trace capture gives 2.8x speedup
- **How:** `jax.make_jaxpr` → walk ops → TT-NN calls → `ttnn.begin_trace_capture`
- **Pros:** Full control, fast iteration, working NOW
- **Cons:** Python dispatch overhead (eliminated by trace), limited ops
- **Missing ops:** gather, scatter, dynamic_update_slice, sine, cosine, sigmoid

## Ambitious: Lightweight PJRT Plugin (jax-mps style)
- **Model:** jax-mps maps StableHLO ops to MLX in ~4 C++ files, 91.5% JAX test coverage
- **Our version:** StableHLO → TT-NN op mapping in C++, bypass tt-mlir entirely
- **Pros:** True `jax.jit` integration, no Python dispatch
- **Cons:** Requires C++ build against XLA, more complex
- **Key insight:** Graph caching in jax-mps is like our trace capture but at the op level

## Official: tt-xla PJRT Plugin
- **Status:** 2,244 commits, active development, depends on tt-mlir compiler
- **Architecture:** JAX → StableHLO → tt-mlir → TT-NN
- **Blackhole support:** Unclear/partial
- **Risk:** Heavy dependency on tt-mlir maturity

## Recommendation
Jaxpr interpreter is pragmatic path. Lightweight PJRT (jax-mps style) is the ambitious goal.
The official tt-xla path has too many dependencies we can't control.
