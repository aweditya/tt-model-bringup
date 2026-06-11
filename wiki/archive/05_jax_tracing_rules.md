# The Rules of JAX: Why Pure Functions Matter

## Q: Why does JAX require pure functions?

**A:** Because the **tracer must produce a complete, correct graph** of your computation. If your function has side effects, mutation, or data-dependent Python control flow, the traced graph won't match what the function actually does.

Pure functions guarantee:
- Same inputs → same outputs (enables caching compiled code)
- No side effects (enables reordering, fusion)
- No mutation (enables differentiation, parallelism)

## Q: What happens if I use Python `if` inside `jax.jit`?

**A (Experiment 05):** JAX catches it and raises `TracerBoolConversionError`.

```python
@jax.jit
def abs_value(x):
    if x > 0:    # ← FAILS: can't convert tracer to bool
        return x
    else:
        return -x
```

During tracing, `x` is not a real number — it's a **tracer** (an abstract placeholder recording what operations happen). Python's `if` tries to call `bool(x)` on the tracer, which is meaningless. JAX refuses rather than silently producing wrong code.

**Fix:** Use `jax.lax.cond`, which traces **both branches** and emits a `stablehlo.case` op:

```python
def abs_value(x):
    return jax.lax.cond(x > 0, lambda: x, lambda: -x)
```

The StableHLO includes both branches:
```mlir
%0 = stablehlo.compare GT, %arg0, %cst
"stablehlo.case"(%0) ({
    %3 = stablehlo.negate %arg0    // false branch
}, {
    stablehlo.return %arg0         // true branch
})
```

## Q: What about Python loops?

**A (Experiment 05):**

**Fixed loops** (`for i in range(5)`) — **OK**. They get unrolled at trace time. The jaxpr just contains 5 copies of the body.

**Data-dependent loops** (`while x < 100`) — **FAILS** with `TracerBoolConversionError`. Same reason as `if`.

**Fix:** Use `jax.lax.while_loop(cond_fn, body_fn, init_val)`:
```python
jax.lax.while_loop(lambda x: x < 100, lambda x: x * 2, init_x)
# Result: 128.0 (starting from 1.0)
```

## Q: What happens to side effects inside jit?

**A (Experiment 05):** They run **once during tracing**, then **never again**.

```python
call_count = 0

@jax.jit
def fn(x):
    global call_count
    call_count += 1    # only executes during tracing!
    return x * 2

fn(3.0)  # call_count = 1 (traced)
fn(5.0)  # call_count = 1 (cached binary, no tracing)
fn(7.0)  # call_count = 1 (cached binary, no tracing)
```

This is subtle and dangerous. `print()` inside jit also only prints during tracing. Use `jax.debug.print()` for runtime printing.

## Q: Why are JAX arrays immutable?

**A (Experiment 05):** Mutation breaks tracing because the tracer doesn't execute real operations — it only records them. If tracing records "write arr[0] = 99", it needs to know what arr[0] was before. With mutation, the trace would depend on operation **order**, not just the computation graph. This makes reordering and fusion impossible.

```python
# NumPy:  arr[0] = 99    ← works (mutation)
# JAX:    arr[0] = 99    ← TypeError!
# JAX:    arr.at[0].set(99)  ← returns NEW array (functional update)
```

## Q: Why does JAX have explicit random keys?

**A (Experiment 05):** Global RNG state is a side effect. Same key = same result, always:

```python
key = jax.random.PRNGKey(42)
jax.random.normal(key, (3,))  # [-0.028, 0.467, 0.296]
jax.random.normal(key, (3,))  # [-0.028, 0.467, 0.296]  ← SAME

k1, k2 = jax.random.split(key)  # create new independent keys
jax.random.normal(k1, (3,))  # [0.076, -0.486, 1.290]  ← different
```

This enables reproducibility and parallelism (independent keys on different devices).

## The Five Rules

| Rule | Why | Fix |
|------|-----|-----|
| No Python `if`/`while` on traced values | Tracer can't evaluate conditions | `jax.lax.cond`, `jax.lax.while_loop` |
| No side effects | Only run during trace, not execution | `jax.debug.print`, return values instead |
| No in-place mutation | Breaks graph reordering and fusion | `x.at[i].set(v)` functional updates |
| No implicit random state | Global state is a side effect | Explicit `PRNGKey`, `split` |
| Functions must be pure | Enables compile, grad, vmap, pmap | Design around transformations |

**Why endure these restrictions?** They guarantee the compiler gets a *complete, correct* graph. Which enables:
- **Reordering** (no mutation order dependencies)
- **Fusion** (no side effects between ops)
- **Differentiation** (pure functions have clean derivatives)
- **Parallelism** (no shared mutable state)

## Experiment

`experiments/05_tracing_gotchas.py` — tests all 6 gotchas with concrete examples.

## Sources
- Experiment 05 results (run 2026-04-21 on CPU)
- JAX sharp bits: https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html
