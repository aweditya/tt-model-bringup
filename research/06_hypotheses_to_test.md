# Hypotheses to Test

## H1: "TT-XLA is really slow"

**Claim**: TT-XLA is slow compared to native TT-NN implementations.

**What we know so far**:
- tt-xla has 3 optimization levels (0, 1, 2) with very different performance
- ResNet-50 via tt-xla achieves ~800 FPS (vs ~220 FPS via tt-forge-onnx — so tt-xla is actually faster than ONNX path)
- Performance docs emphasize warmup (3+ iterations), runtime tracing, and data format as critical
- 853 open issues suggests active development but also many rough edges
- No public head-to-head benchmark of tt-xla vs hand-tuned TT-NN for the same model

**Experiment plan**:
1. Install tt-xla on our Blackhole device
2. Run a simple model (e.g., matrix multiply, then a small transformer) through tt-xla
3. Run the same operation directly via TT-NN/Metalium
4. Compare: latency, throughput, compilation time
5. Vary optimization levels (0, 1, 2) and measure each

**What "slow" could mean**:
- Compilation time is slow (XLA compile + MLIR lowering overhead)
- Runtime is slow (suboptimal code generation, missing fusion passes)
- Both
- Or maybe it's actually fine and the claim is outdated?

## H2: "Writing MLIR passes is really hard"

**Claim**: Writing custom optimization passes for MLIR is extremely difficult.

**What we know so far**:
- MLIR is a general compiler framework with dialects, passes, and progressive lowering
- tt-mlir has its own dialects: TTIR (hardware-agnostic) and TTNN (backend-specific)
- Writing passes requires understanding the IR structure, pattern matching, and transformation rules
- The MLIR ecosystem has tools: tablegen for op definitions, pattern rewrite framework, pass infrastructure

**Experiment plan**:
1. Read through tt-mlir source to understand existing passes
2. Find a simple optimization that's missing (check GitHub issues)
3. Attempt to write a minimal pass
4. Document the difficulty honestly

**What "hard" could mean**:
- Steep learning curve (MLIR concepts are novel)
- Debugging is painful (IR dumps are huge, error messages cryptic)
- Correctness is hard (easy to introduce subtle bugs)
- All of the above
- Or maybe the tooling has improved and it's tractable?

## H3: JAX compilation overhead is worth it for Tenstorrent

**Claim**: The whole-program optimization that JAX/XLA enables should be especially beneficial for Tenstorrent's architecture.

**Reasoning**:
- Tenstorrent has no cache hierarchy — data movement is explicit and expensive
- XLA's fusion passes can eliminate intermediate memory round-trips
- The 3-kernel structure (reader/compute/writer) maps naturally to a dataflow graph
- JAX's functional paradigm gives the compiler maximum visibility

**Experiment plan**:
1. Take a multi-op computation (e.g., linear + relu + linear)
2. Run unfused (separate TT-NN calls) vs fused (single tt-xla compilation)
3. Measure DRAM traffic and latency difference
4. This tests whether the compiler fusion actually reduces data movement

## H4: The codegen path is more useful than JIT for Tenstorrent

**Claim**: TT-XLA's codegen (generating standalone C++/Python calling TT-NN) may be more practical than JIT compilation.

**Reasoning**:
- JIT has compilation overhead on every new input shape
- Codegen produces readable, debuggable code
- Generated code can be hand-optimized after generation
- For production, you want a compiled artifact, not JIT

**Experiment plan**:
1. Use tt-xla codegen to generate C++ for a model
2. Inspect the generated code — is it readable? Is it what a human would write?
3. Compare performance of codegen output vs JIT execution
4. Try hand-optimizing the generated code and measure improvement

## Priority Order

1. **H1** (is tt-xla slow?) — most impactful, directly actionable
2. **H3** (does XLA fusion help on TT?) — tests the core thesis of this project
3. **H4** (codegen vs JIT) — practical for future work
4. **H2** (MLIR passes are hard) — informational, less urgent
