"""
Experiment 05: JAX Tracing Gotchas & Functional Programming
============================================================
Hypothesis: JAX tracing captures only one branch of Python if/else,
so JIT'd functions with data-dependent control flow silently give
wrong results. Pure functions are required.

We test:
  1. Python if inside jit — does it break?
  2. jax.lax.cond — the correct alternative
  3. Side effects inside jit — what happens?
  4. In-place mutation — why JAX forbids it
  5. Closures over mutable state — subtle bug
"""

import jax
import jax.numpy as jnp

# ============================================================
# TEST 1: Python if/else inside jit
# ============================================================
print("=" * 60)
print("TEST 1: Python if/else inside jit")
print("=" * 60)

def abs_value_python_if(x):
    """Uses Python if — BROKEN under jit."""
    if x > 0:
        return x
    else:
        return -x

# Without jit: works fine
print(f"\n  Without jit:")
print(f"    abs_value(5.0)  = {abs_value_python_if(5.0)}")
print(f"    abs_value(-3.0) = {abs_value_python_if(-3.0)}")

# With jit: traces with first input, bakes in that branch
print(f"\n  With jit (first call with 5.0, positive branch traced):")
abs_jit = jax.jit(abs_value_python_if)
try:
    result_pos = abs_jit(5.0)
    print(f"    abs_jit(5.0)  = {result_pos}")
    result_neg = abs_jit(-3.0)
    print(f"    abs_jit(-3.0) = {result_neg}  ← WRONG! Should be 3.0!")
    print(f"    BUG: traced the 'if x > 0' branch with abstract value,")
    print(f"    which raised ConcretizationTypeError or baked in one branch.")
except jax.errors.TracerBoolConversionError as e:
    print(f"    JAX caught it! Error: {type(e).__name__}")
    print(f"    Message: {str(e)[:200]}")
    print(f"\n    JAX prevents you from using Python if on traced values.")
    print(f"    This is a SAFETY feature — it refuses rather than silently breaking.")
except Exception as e:
    print(f"    Error: {type(e).__name__}: {str(e)[:200]}")

# ============================================================
# TEST 2: jax.lax.cond — the correct way
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 2: jax.lax.cond — data-dependent branching done right")
print("=" * 60)

def abs_value_lax(x):
    """Uses jax.lax.cond — works correctly under jit."""
    return jax.lax.cond(
        x > 0,
        lambda: x,       # true branch
        lambda: -x,      # false branch
    )

abs_lax_jit = jax.jit(abs_value_lax)
print(f"\n  abs_lax_jit(5.0)  = {abs_lax_jit(5.0)}")
print(f"  abs_lax_jit(-3.0) = {abs_lax_jit(-3.0)}")
print(f"  Correct! Both branches are traced and included in the compiled code.")

# Show the HLO — both branches are there
lowered = jax.jit(abs_value_lax).lower(jnp.float32(1.0))
hlo = lowered.as_text()
has_conditional = "stablehlo.if" in hlo or "conditional" in hlo.lower() or "select" in hlo.lower()
print(f"\n  StableHLO contains conditional/select: {has_conditional}")
print(f"  HLO ({len(hlo)} chars):")
print(f"  {hlo[:600]}")

# ============================================================
# TEST 3: jax.lax.while_loop vs Python while
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 3: Loops inside jit")
print("=" * 60)

# Python for with FIXED bounds: OK (loop is unrolled at trace time)
@jax.jit
def fixed_loop(x):
    for i in range(5):  # fixed — unrolled during tracing
        x = x * 2
    return x

print(f"\n  Fixed Python loop (range(5)): {fixed_loop(jnp.float32(1.0))}")
print(f"  Works! The loop is unrolled: becomes x*2*2*2*2*2 = x*32")

# Python while with data-dependent condition: BROKEN
def data_while(x):
    while x < 100:
        x = x * 2
    return x

print(f"\n  Data-dependent while loop:")
try:
    jax.jit(data_while)(jnp.float32(1.0))
except Exception as e:
    print(f"    Error: {type(e).__name__}")
    print(f"    Can't use Python while on traced values.")

# jax.lax.while_loop: the correct way
def data_while_lax(x):
    return jax.lax.while_loop(
        lambda x: x < 100,    # condition
        lambda x: x * 2,      # body
        x                      # initial value
    )

print(f"\n  jax.lax.while_loop: {jax.jit(data_while_lax)(jnp.float32(1.0))}")
print(f"  Correct! Loop condition checked at runtime, not trace time.")

# ============================================================
# TEST 4: Side effects inside jit
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 4: Side effects inside jit")
print("=" * 60)

call_count = 0

@jax.jit
def fn_with_side_effect(x):
    global call_count
    call_count += 1  # side effect!
    return x * 2

print(f"\n  call_count before: {call_count}")
result = fn_with_side_effect(jnp.float32(3.0))
print(f"  call_count after 1st call: {call_count}")
result = fn_with_side_effect(jnp.float32(5.0))
print(f"  call_count after 2nd call: {call_count}")
result = fn_with_side_effect(jnp.float32(7.0))
print(f"  call_count after 3rd call: {call_count}")
print(f"""
  Side effects only execute during TRACING, not during cached execution.
  call_count incremented once (during trace), then never again.
  This is why JAX requires PURE functions — no side effects!
""")

# ============================================================
# TEST 5: Mutation — why JAX arrays are immutable
# ============================================================
print(f"{'=' * 60}")
print("TEST 5: Why JAX arrays are immutable")
print("=" * 60)

# NumPy: mutation works
import numpy as np
np_arr = np.array([1.0, 2.0, 3.0])
np_arr[0] = 99.0
print(f"\n  NumPy mutation:  {np_arr}  ← works, arr[0] = 99")

# JAX: mutation is forbidden
jax_arr = jnp.array([1.0, 2.0, 3.0])
try:
    jax_arr[0] = 99.0
except TypeError as e:
    print(f"  JAX mutation:    TypeError: {str(e)[:80]}")

# Why? Because mutation breaks tracing:
print(f"""
  If tracing records 'write arr[0] = 99', what value did arr[0] have before?
  The tracer doesn't execute real operations — it only records them.
  With mutation, the trace would depend on ORDER of operations, not just
  the computation graph. This makes optimization (reordering, fusion) impossible.

  Instead, JAX uses functional updates:""")

jax_arr_new = jax_arr.at[0].set(99.0)
print(f"  jax_arr.at[0].set(99) = {jax_arr_new}  (new array, original unchanged)")
print(f"  original:               {jax_arr}")

# ============================================================
# TEST 6: Random numbers — also require explicit state
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 6: Random numbers require explicit keys")
print("=" * 60)

# NumPy: global state, implicit
print(f"\n  NumPy random (global state):")
print(f"    {np.random.randn(3)}")
print(f"    {np.random.randn(3)}  ← different (global state mutated)")

# JAX: explicit key, deterministic
print(f"\n  JAX random (explicit key):")
key = jax.random.PRNGKey(42)
print(f"    key={key}: {jax.random.normal(key, (3,))}")
print(f"    key={key}: {jax.random.normal(key, (3,))}  ← SAME (same key = same result)")

k1, k2 = jax.random.split(key)
print(f"    k1={k1}: {jax.random.normal(k1, (3,))}")
print(f"    k2={k2}: {jax.random.normal(k2, (3,))}  ← different (different key)")

print(f"""
  JAX requires explicit random state because:
  1. Global RNG state is a side effect (violates purity)
  2. Reproducibility: same key = same result, always
  3. Parallelism: independent keys can run on different devices
  4. JIT: traced function must be deterministic for given inputs
""")

# ============================================================
# SUMMARY
# ============================================================
print("=" * 60)
print("SUMMARY: The Rules of JAX")
print("=" * 60)
print("""
  1. NO Python control flow on traced values
     → Use jax.lax.cond, jax.lax.while_loop, jax.lax.scan

  2. NO side effects (print, global mutation, I/O)
     �� Side effects only run during tracing, not execution

  3. NO in-place mutation of arrays
     → Use x.at[i].set(v) for functional updates

  4. NO implicit random state
     → Use explicit jax.random.PRNGKey, split for new keys

  5. Functions must be PURE: same inputs → same outputs, no side effects
     → This is what makes tracing, compilation, and grad possible

  WHY these restrictions?
  They guarantee the traced jaxpr is a COMPLETE, CORRECT description
  of the computation. The compiler can then freely:
    - Reorder operations (no mutation ordering dependencies)
    - Fuse operations (no side effects between them)
    - Differentiate (pure functions have clean derivatives)
    - Parallelize (no shared mutable state)
""")
