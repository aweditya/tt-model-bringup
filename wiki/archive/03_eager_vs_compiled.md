# Eager vs Compiled Execution: PyTorch vs JAX

## Q: What is "eager execution"?

**A:** Eager execution means each operation runs immediately, one at a time, as Python encounters it. There is no compilation step.

When PyTorch runs `h = torch.relu(x @ w1)`:

```
Step 1: Python evaluates x @ w1
  → Python calls x.__matmul__(w1)
  → PyTorch's C++ dispatcher receives the call
  → Dispatcher checks types, selects the CPU matmul kernel
  → Kernel runs, allocates output tensor in memory
  → Returns result to Python interpreter

Step 2: Python evaluates torch.relu(result)
  → Python calls into torch.relu C++ code
  → Dispatcher selects the CPU relu kernel  
  → Kernel runs, allocates ANOTHER tensor in memory
  → Returns to Python
```

Each operation is **independent**. PyTorch doesn't know relu comes after matmul until it actually happens at runtime. The intermediate tensor (matmul output) gets written to memory, then relu reads it back from memory.

**Advantages**: Zero compilation cost, immediate execution, easy to debug (you can print intermediate values, set breakpoints, use Python control flow freely).

**Disadvantages**: Python interpreter overhead between every op, no fusion (extra memory traffic), no whole-program optimization.

## Q: What is "compiled execution" (JIT)?

**A:** The program is first **traced** into a computation graph, then **compiled** into optimized machine code. Execution happens only after compilation.

When JAX runs `jax.jit(f)(x, w1)`:

```
First call:
  1. Trace: run f with abstract values, record ops → Jaxpr
  2. Lower: Jaxpr → StableHLO (MLIR)
  3. Optimize: XLA fuses ops, assigns layouts, simplifies
  4. Compile: StableHLO → LLVM IR → machine code
  5. Execute: run the compiled binary

Subsequent calls (same shapes):
  1. Cache lookup → find compiled binary
  2. Execute
```

## Q: How much does compilation actually cost?

**A (Experiment 03, CPU):** For a 2-layer MLP (64×256 → 128 → 64):

| Phase | Time |
|-------|------|
| Tracing (Python → Jaxpr) | 2.4 ms |
| Lowering (Jaxpr → StableHLO) | 2.5 ms |
| Compilation (StableHLO → machine code) | 11.0 ms |
| First execution | 9.0 ms |
| **Total first call** | **25.0 ms** |
| **Each subsequent call** | **0.099 ms** |

**Break-even: ~253 calls.** After 253 calls, the compilation cost has been amortized and you're ahead. For training (thousands of steps) or inference (many requests), this is easily worth it.

Yes — this is exactly what people mean by "JIT is slow the first time." The first call pays for tracing + lowering + compilation. Every subsequent call just runs the cached binary.

## Q: What triggers recompilation?

**A (Experiment 03):** Changing input **shapes** triggers recompilation. Changing **values** does not.

```
Call 1 (shape 64×256, compile):  0.5 ms   ← includes compilation
Call 2 (shape 64×256, cached):   0.1 ms   ← cache hit
Call 3 (shape 32×256, compile):  15.5 ms  ← NEW SHAPE → recompile!
Call 4 (shape 32×256, cached):   0.1 ms   ← cache hit
Call 5 (shape 64×256, cached):   0.2 ms   ← original shape still cached
```

JAX maintains a cache keyed by (function, input shapes, input dtypes). Each unique combination gets its own compiled binary. This is why dynamic batch sizes can be painful in JAX — each new batch size triggers a full recompilation.

## Q: Which is faster — PyTorch eager or JAX compiled?

**A (Experiment 03, CPU, 2-layer MLP):**

| Mode | Time per call | First-call cost |
|------|--------------|-----------------|
| PyTorch eager | 0.032 ms | ~0 ms |
| JAX compiled | 0.099 ms | 25 ms |
| torch.compile | 0.066 ms | **7,417 ms** |

**Surprise: on CPU with small tensors, PyTorch eager is faster!**

Why? For a 2-layer MLP on CPU:
- The matmuls are already calling optimized BLAS (MKL/OpenBLAS)
- The computation is small enough that dispatch overhead doesn't matter much
- JAX's compiled code calls the same BLAS underneath
- JAX has overhead from its runtime (buffer management, etc.)

**This does NOT mean eager is always faster.** The advantage of compilation grows with:
1. **More operations** — more Python overhead to eliminate, more fusion opportunities
2. **Accelerators (GPU/TPU/Tenstorrent)** — kernel launch overhead is much larger than CPU dispatch
3. **Larger models** — fusion saves memory bandwidth, which is the real bottleneck

The rvLLM result (16k tok/s on TPU) works because the entire forward pass is one fused program — eliminating hundreds of kernel launches per forward pass.

## Q: What about torch.compile?

**A:** `torch.compile` is PyTorch's answer to JAX's JIT. It traces the computation graph and compiles it. But notice:

- **Compilation time: 7,417 ms** (7.4 seconds!) — much slower than JAX's 25ms because torch.compile uses Dynamo (Python bytecode analysis) + Inductor (code generation), which is a heavier pipeline
- **Subsequent calls: 0.066 ms** — faster than PyTorch eager, comparable to JAX

torch.compile is a newer, more complex compilation strategy. JAX was designed for compilation from the start; PyTorch retrofitted it.

## Q: What about per-operation dispatch overhead?

**A (Experiment 03):** For `relu(float[4])` — a tiny operation where the actual compute is negligible:

```
PyTorch dispatch: 0.9 µs
JAX no-jit:       7.4 µs
JAX jit:          3.2 µs
```

PyTorch's C++ dispatcher is highly optimized (~1µs). JAX without jit goes through more Python-level machinery (~7µs). JAX with jit still has runtime overhead (~3µs) for cache lookup + buffer management.

**Lesson**: For tiny operations, eager dispatch is cheaper than compiled dispatch. Compilation wins when you can amortize over many operations fused together.

## Q: What is a Jaxpr?

**A:** A jaxpr (JAX expression) is JAX's internal intermediate representation — the output of tracing. It's a flat list of typed operations:

```
{ lambda ; a:f32[4,3] b:f32[3,2]. let
    c:f32[4,2] = dot_general a b
    d:f32[4,2] = max c 0.0
  in (d,) }
```

Reading it:
- `lambda ; a:f32[4,3] b:f32[3,2]` — inputs with types
- `c:f32[4,2] = dot_general a b` — operations with named results
- `in (d,)` — outputs

It's simpler than StableHLO (no broadcast semantics, no MLIR syntax). Think of it as JAX's "notes" that get cleaned up into proper StableHLO for the compiler.

The key property: **a jaxpr has no Python control flow**. When JAX traces `if x > 0: ...`, it evaluates the condition with the tracer value and only records the branch that was taken. This is why JIT'd functions must be careful with Python conditionals — use `jax.lax.cond` instead of `if` for data-dependent branching.

## The Big Picture

```
                    EAGER                          COMPILED
                    ─────                          ────────
Python code    →  Execute each op immediately  →  Trace → Compile → Execute binary
                                                  
Per-op cost:      ~1µs (dispatch)                 ~0 (fused into binary)
First call:       instant                         slow (compilation)
Memory:           intermediate tensors in RAM     fused (fewer intermediates)
Debug:            easy (print, breakpoint)         harder (can't step into HLO)
Dynamic shapes:   free                            recompilation per shape

Best for:         debugging, prototyping,         training loops, inference,
                  dynamic input sizes             accelerators, large models
```

## Experiment

Run `experiments/03_jit_overhead_and_eager.py`:
```bash
ssh tenstorrent
source ~/tt-xla-env/bin/activate
python3 ~/03_jit_overhead_and_eager.py
```

## Sources
- Experiment 03 results (run 2026-04-21 on CPU)
- JAX JIT docs: https://jax.readthedocs.io/en/latest/jit-compilation.html
