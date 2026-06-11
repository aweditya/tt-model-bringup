# Dialects, Backends, and What "Interpretation" Actually Means

## Q: Why StableHLO instead of XLA's HLO? Why are there so many dialects?

**A:** History and a real engineering problem.

### The History

Originally there was just **HLO** — XLA's internal representation. It lived inside TensorFlow/XLA and changed whenever Google engineers needed to. This was fine when XLA was only used internally.

Then other frameworks (JAX, PyTorch) and compilers (IREE, TT-MLIR) wanted to use it. Problem: HLO had **no stability guarantees**. If Google changed an op's semantics, every downstream project broke.

The timeline:
1. **HLO** (original) — XLA's internal IR, unstable, changes freely
2. **MHLO** (2020) — "MLIR-HLO", an MLIR dialect mirroring HLO ops. Still unstable.
3. **StableHLO** (2022) — A fork of MHLO with **5-year backward compatibility**. The "stable" contract.
4. **VHLO** — "Versioned HLO", used for serialization. Snapshots StableHLO at each version so old binaries still work.

### Why StableHLO Won

If you're Tenstorrent building a compiler, you don't want it to break every time Google updates XLA. StableHLO guarantees your compiler built against version X will still work 5 years later. That's why tt-mlir consumes StableHLO, not raw HLO.

### Why So Many MLIR Dialects?

The core idea is **"premature lowering is the root of all evil"** (actual quote from the MLIR paper).

Consider compiling `y = relu(matmul(x, w))`:

- At the **StableHLO level**: you see `dot_general` and `maximum` — high-level tensor ops. The compiler can reason about shapes, fuse operations, assign memory layouts.
- At the **LLVM IR level**: you see `fmul`, `fadd`, loops, pointer arithmetic. The matmul is 50 lines of loop nests. Good luck trying to fuse the relu now — you'd have to reverse-engineer that those loops ARE a matmul.

**Each dialect preserves the right semantics for its level of optimization:**

```
StableHLO:  "matmul these tensors, then relu"
            → Can fuse! Can change layout! Can swap for library call!

LLVM IR:    "loop i=0..64: loop j=0..128: loop k=0..256: fma..."
            → Can register-allocate. Can vectorize. Can't unfuse.
```

If you lower to LLVM IR too early, you destroy the information needed for high-level optimizations. Multiple dialects let you optimize at each level before dropping to the next.

For Tenstorrent specifically:
```
StableHLO  →  "matmul + relu on tensors"     (can decide: fuse? shard? which cores?)
TTIR       →  "matmul + relu on TT tiles"    (can decide: data layout? L1 vs DRAM?)
TTNN       →  "ttnn::matmul() + ttnn::relu()" (maps to actual TT-NN API calls)
Metalium   →  reader/compute/writer kernels   (actual hardware instructions)
```

Each arrow is a **lowering pass** that makes decisions appropriate for that abstraction level.

## Q: For TT, the dialect maps IR ops to hand-written Metalium kernels. Is this the best approach?

**A:** It's a pragmatic tradeoff. Let's compare how three backends do it:

### Approach 1: NVIDIA GPU (XLA:GPU) — Hybrid: Library Calls + Code Generation

XLA:GPU uses **three strategies** depending on the operation:

| Strategy | Used For | How It Works |
|----------|----------|-------------|
| **Library calls** | matmul, conv, FFT | Calls cuBLAS, cuDNN, NCCL — hand-tuned by NVIDIA |
| **LLVM codegen** | reductions, transposes, elementwise | Generates LLVM IR → PTX → SASS from scratch |
| **Triton codegen** | fused matmul+softmax, complex fusions | Generates TritonIR → PTX, auto-tunes tile sizes |

The key insight: **matmul is a library call, but the ops fused around it are generated code.** XLA generates the "glue" kernels that fuse relu, add, scale, etc. around the matmul library call.

This is why our experiment showed the HLO as:
```
%dot.6 = dot(...)                                    ← library call (BLAS)
%broadcast_maximum_fusion = fusion(%dot.6, %bias)    ← GENERATED kernel (add+relu fused)
```

### Approach 2: Google TPU — Full Code Generation

For TPUs, XLA goes further: HLO → **LLO** (Low-Level Optimized) → TPU instructions. Google wrote the entire backend because **they designed both the compiler and the hardware.** No need for library calls when you control the silicon.

The TPU backend can fuse more aggressively because it knows exactly what the hardware can execute. This is why JAX on TPU often outperforms JAX on GPU — tighter compiler-hardware co-design.

### Approach 3: Tenstorrent (TT-MLIR) — Map to Pre-Written Kernels

TT-MLIR's approach: lower StableHLO → TTIR → TTNN, where TTNN operations map 1:1 to TT-NN library functions. Those functions invoke hand-written Metalium kernels (reader/compute/writer triplets).

```
stablehlo.dot_general  →  ttnn::matmul()   →  hand-written Metalium kernel
stablehlo.maximum      →  ttnn::relu()      →  hand-written Metalium kernel
```

**Pros:**
- Kernels are hand-optimized by Tenstorrent engineers who know the hardware intimately
- Faster to bring up — don't need to write a full code generator
- Each kernel can exploit hardware-specific tricks (tile sizes, data movement patterns)

**Cons:**
- **Fusion is limited** — you can only fuse ops that have a pre-written fused kernel
- If TT-NN doesn't have a `matmul_relu_add` fused kernel, those remain 3 separate dispatches
- Unlike XLA:GPU which *generates* the fused glue kernels, TT-MLIR can only select from what exists
- Adding new fusions means writing new Metalium kernels by hand

**Is this the best approach?** For now, it's the practical one. Tenstorrent is a small team compared to Google/NVIDIA. Writing a full code generator (like XLA:GPU's LLVM emitters or Triton integration) is a massive investment. Mapping to hand-written kernels gets you working hardware support faster.

The long-term ideal would be a code generator that can emit arbitrary fused kernels for Tensix — like how Triton generates arbitrary fused GPU kernels. But that requires deep understanding of the Tensix ISA, register file, memory hierarchy, and pipeline scheduling. Much harder than calling pre-written kernels.

## Q: What exactly is "interpretation"?

**A:** Interpretation means **walking a computation graph node by node, executing each operation independently.**

### Concrete example

Say we have the graph: `y = relu(x @ w + b)`

As three nodes:
```
node 1: tmp1 = matmul(x, w)
node 2: tmp2 = add(tmp1, b)
node 3: y    = relu(tmp2)
```

**Interpretation (what PyTorch eager does):**
```python
for node in graph:
    inputs = look_up(node.inputs)  # get tensors from memory
    result = dispatch(node.op, inputs)  # call the kernel
    store(node.name, result)  # write result to memory
```

Each `dispatch` is a separate event:
1. Python → C++ FFI boundary crossing
2. Dispatcher: check types, find the right kernel implementation
3. Kernel: allocate output memory, compute, write result
4. Return to Python, ready for next node

We built a literal interpreter in Experiment 04 — it's just a for-loop over ops:
```python
def interpret(graph, env):
    for op in graph:
        args = [env[inp] for inp in op.inputs]
        env[op.name] = op.fn(*args)
```

**Compilation (what JAX does):**

Instead of walking the graph at runtime, transform the entire graph into a single optimized program *before* running it:
```
matmul(x, w) → add(_, b) → relu(_)
                    ↓ XLA fusion
           dot(x, w) → fused_add_relu(_, b)
                    ↓ code generation
           single binary (machine code)
```

At runtime: call the binary. No graph walking, no dispatch, no intermediate memory allocation.

### The TorchScript graph makes this visible

Our experiment captured PyTorch's computation graph:
```
graph(%x, %w, %b):
  %3 = aten::matmul(%x, %w)
  %5 = aten::add(%3, %b, %4)
  %6 = aten::relu(%5)
  return (%6)
```

When PyTorch runs this eagerly: it interprets this graph — executing `aten::matmul`, then `aten::add`, then `aten::relu`, each as a separate C++ function call.

When `torch.compile` processes this: it transforms the graph, fuses what it can, and compiles to a single executable. Same idea as JAX, different implementation.

### Performance from Experiment 04

```
PyTorch eager:          50.8 µs    (interpret each op)
torch.jit.trace:        19.0 µs    (optimized graph, less dispatch overhead)
torch.compile:          46.7 µs    (compiled, but Inductor overhead on CPU)
JAX jit (from exp 03):   0.099 ms   (XLA compiled)
```

On CPU with small tensors, the differences are modest because the actual compute (BLAS matmul) dominates. The dramatic wins come on accelerators where:
- Kernel launch overhead is 5-20µs *per op* (vs ~1µs on CPU)
- Memory bandwidth is the bottleneck (fusion eliminates intermediate writes to HBM)
- A 100-layer transformer has hundreds of ops to fuse

## Q: How does fusion help concretely?

**A (Experiment 04):** For a 3-layer chain with 9 operations (3 matmuls + 3 relus + 3 scale/shifts):

```
Unfused (9 separate ops): 1.886 ms
Fused (jit compiled):     1.582 ms    ← 1.2x speedup, CPU only
```

XLA produced: **3 dot ops + 3 fusion ops** (relu+scale+shift fused around each matmul).

Only 1.2x on CPU because the matmuls dominate compute time and fusion only saves memory traffic on the small elementwise ops. On GPU where memory bandwidth is the bottleneck, fusion of elementwise ops around matmuls regularly gives **2-5x** speedup.

## The Full Picture: What Runs When

```
                     PyTorch Eager          JAX Compiled         TT-XLA
                     ─────────────          ────────────         ──────
Python says:         x @ w                  jax.jit(f)(x, w)    jax.jit(f)(x, w)

What happens:        Python→C++ dispatch    [first time only:    [first time only:
                     → find matmul kernel    trace→jaxpr          trace→jaxpr
                     → BLAS call             →StableHLO           →StableHLO
                     → return tensor         →HLO optimize        →TTIR
                     → next op...            →LLVM IR→x86]        →TTNN
                                                                  →Metalium kernels]
                     then relu dispatch      Run cached binary    Run cached binary
                     → find relu kernel
                     → elementwise pass
                     → return tensor

Memory traffic:      matmul writes to RAM   matmul writes to     matmul writes to
                     relu reads from RAM     cache/registers,     L1 SRAM,
                     relu writes to RAM      fused relu in same   fused relu in same
                                             pass                 pass (if kernel exists)

Per-call overhead:   ~1µs dispatch × N ops  ~0 (one binary)      ~0 (one binary)
```

## Experiment

Run `experiments/04_interpretation_vs_compilation.py`:
```bash
ssh tenstorrent
source ~/tt-xla-env/bin/activate
python3 ~/04_interpretation_vs_compilation.py
```

## Sources
- Experiment 04 results (run 2026-04-21 on CPU)
- XLA GPU architecture: https://openxla.org/xla/gpu_architecture
- XLA architecture: https://openxla.org/xla/architecture
- StableHLO history: https://github.com/openxla/stablehlo
- MLIR paper: "MLIR: Scaling Compiler Infrastructure for Domain Specific Computation"
- MLIR progressive lowering: https://www.cs.cornell.edu/courses/cs6120/2023fa/blog/mlir/
