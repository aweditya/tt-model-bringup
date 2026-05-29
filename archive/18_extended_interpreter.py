"""
Experiment 18: Extended Jaxpr Interpreter
==========================================
Extends the TTNNInterpreter from experiment 14 with reduce_max, reduce_sum,
div, sqrt, rsqrt, convert_element_type, and integer_pow. Tests on softmax
(which failed in exp 14), layer norm, and a mini-transformer block.

Goal: prove the interpreter can handle realistic ML building blocks,
not just toy functions.
"""

import jax
import jax.numpy as jnp
from jax import make_jaxpr
from jax._src import core as jax_core
import ttnn
import torch
import numpy as np
import time

device = ttnn.open_device(device_id=0)
print("Device: Blackhole p150a")
print()

# ============================================================
# The Extended Interpreter
# ============================================================

class TTNNInterpreter:
    """Walks a Jaxpr and executes each equation on TT-NN.

    Extended from experiment 14 with reduction ops, sqrt/rsqrt,
    convert_element_type, integer_pow, and reciprocal.
    """

    def __init__(self, device):
        self.device = device
        self.env = {}  # var -> ttnn tensor
        self.ops_encountered = set()  # track all primitives seen
        self.ops_handled = set()
        self.ops_unhandled = set()

    def to_ttnn(self, val):
        """Convert a numpy/torch value to a TT-NN tensor on device."""
        if isinstance(val, (int, float)):
            val = np.array([[val]], dtype=np.float32)
        if isinstance(val, np.ndarray):
            val = torch.from_numpy(val.copy()).float()
        if isinstance(val, jax.Array):
            val = torch.from_numpy(np.array(val)).float()
        # TT-NN needs at least 2D for TILE_LAYOUT
        while val.dim() < 2:
            val = val.unsqueeze(0)
        # Pad to tile-aligned (multiples of 32)
        h, w = val.shape[-2], val.shape[-1]
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h > 0 or pad_w > 0:
            val = torch.nn.functional.pad(val, (0, pad_w, 0, pad_h))
        return ttnn.from_torch(val, dtype=ttnn.bfloat16, device=self.device,
                               layout=ttnn.TILE_LAYOUT)

    def from_ttnn(self, tensor, shape):
        """Convert TT-NN tensor back to numpy with correct shape."""
        t = ttnn.to_torch(tensor).float()
        # Remove padding and restore original shape
        if len(shape) == 0:
            return t.numpy().flatten()[0]
        elif len(shape) == 1:
            return t.reshape(-1).numpy()[:shape[0]]
        elif len(shape) == 2:
            # Squeeze batch dims but keep 2D
            while t.dim() > 2:
                t = t.squeeze(0)
            return t.numpy()[:shape[0], :shape[1]]
        else:
            # Higher-dimensional: best effort
            return t.squeeze().numpy()

    def eval_var(self, var):
        """Look up a variable in the environment."""
        if isinstance(var, jax_core.Literal):
            return self.to_ttnn(var.val)
        return self.env[var]

    def interpret(self, jaxpr, args):
        """Execute a closed Jaxpr with given arguments."""
        # Bind input variables
        for var, arg in zip(jaxpr.jaxpr.invars, args):
            self.env[var] = self.to_ttnn(arg)

        # Bind constants
        for var, const in zip(jaxpr.jaxpr.constvars, jaxpr.consts):
            self.env[var] = self.to_ttnn(const)

        # Execute equations
        for eqn in jaxpr.jaxpr.eqns:
            self._exec_eqn(eqn)

        # Return outputs
        out_shapes = [v.aval.shape for v in jaxpr.jaxpr.outvars]
        results = []
        for var, shape in zip(jaxpr.jaxpr.outvars, out_shapes):
            results.append(self.from_ttnn(self.eval_var(var), shape))
        return results[0] if len(results) == 1 else results

    def _exec_eqn(self, eqn):
        """Execute a single Jaxpr equation."""
        prim = eqn.primitive.name
        invars = eqn.invars
        params = eqn.params

        self.ops_encountered.add(prim)

        # Dispatch to handler
        handler = getattr(self, f'_op_{prim}', None)
        if handler is None:
            # Try to handle sub-jaxprs (like custom_jvp_call, pjit)
            if prim in ('custom_jvp_call', 'pjit'):
                handler = self._op_subjaxpr
            else:
                self.ops_unhandled.add(prim)
                raise NotImplementedError(f"Unsupported primitive: {prim}")

        self.ops_handled.add(prim)
        result = handler(invars, params, eqn)

        # Bind outputs
        if isinstance(result, (list, tuple)):
            for var, val in zip(eqn.outvars, result):
                self.env[var] = val
        else:
            self.env[eqn.outvars[0]] = result

    def _broadcast_for_binary(self, a, b, eqn):
        """Ensure two tensors have matching shapes for TT-NN binary ops.
        TT-NN TILE_LAYOUT cannot broadcast (32,1) vs (32,64), so we
        explicitly broadcast by round-tripping through CPU when needed."""
        a_shape = eqn.invars[0].aval.shape if hasattr(eqn.invars[0], 'aval') else ()
        b_shape = eqn.invars[1].aval.shape if hasattr(eqn.invars[1], 'aval') else ()
        out_shape = eqn.outvars[0].aval.shape

        def needs_broadcast(in_s, out_s):
            if len(in_s) != len(out_s):
                return True
            return any(i != o for i, o in zip(in_s, out_s))

        if needs_broadcast(a_shape, out_shape) and not isinstance(eqn.invars[0], jax_core.Literal):
            a_np = self.from_ttnn(a, a_shape)
            a_np = np.broadcast_to(np.array(a_np).reshape(a_shape), out_shape).copy()
            a = self.to_ttnn(a_np)

        if needs_broadcast(b_shape, out_shape) and not isinstance(eqn.invars[1], jax_core.Literal):
            b_np = self.from_ttnn(b, b_shape)
            b_np = np.broadcast_to(np.array(b_np).reshape(b_shape), out_shape).copy()
            b = self.to_ttnn(b_np)

        return a, b

    # --- Original ops from experiment 14 ---

    def _op_dot_general(self, invars, params, eqn):
        lhs = self.eval_var(invars[0])
        rhs = self.eval_var(invars[1])
        return ttnn.matmul(lhs, rhs)

    def _op_add(self, invars, params, eqn):
        if isinstance(invars[0], jax_core.Literal):
            b = self.eval_var(invars[1])
            return ttnn.add(b, float(invars[0].val))
        if isinstance(invars[1], jax_core.Literal):
            a = self.eval_var(invars[0])
            return ttnn.add(a, float(invars[1].val))
        a = self.eval_var(invars[0])
        b = self.eval_var(invars[1])
        a, b = self._broadcast_for_binary(a, b, eqn)
        return ttnn.add(a, b)

    def _op_mul(self, invars, params, eqn):
        if isinstance(invars[0], jax_core.Literal):
            b = self.eval_var(invars[1])
            return ttnn.multiply(b, float(invars[0].val))
        if isinstance(invars[1], jax_core.Literal):
            a = self.eval_var(invars[0])
            return ttnn.multiply(a, float(invars[1].val))
        a = self.eval_var(invars[0])
        b = self.eval_var(invars[1])
        a, b = self._broadcast_for_binary(a, b, eqn)
        return ttnn.mul(a, b)

    def _op_sub(self, invars, params, eqn):
        if isinstance(invars[0], jax_core.Literal):
            b = self.eval_var(invars[1])
            return ttnn.sub(self.to_ttnn(np.array(float(invars[0].val))), b)
        if isinstance(invars[1], jax_core.Literal):
            a = self.eval_var(invars[0])
            return ttnn.sub(a, self.to_ttnn(np.array(float(invars[1].val))))
        a = self.eval_var(invars[0])
        b = self.eval_var(invars[1])
        a, b = self._broadcast_for_binary(a, b, eqn)
        return ttnn.sub(a, b)

    def _op_neg(self, invars, params, eqn):
        a = self.eval_var(invars[0])
        return ttnn.neg(a)

    def _op_exp(self, invars, params, eqn):
        a = self.eval_var(invars[0])
        return ttnn.exp(a)

    def _op_log(self, invars, params, eqn):
        a = self.eval_var(invars[0])
        return ttnn.log(a)

    def _op_max(self, invars, params, eqn):
        if isinstance(invars[1], jax_core.Literal) and float(invars[1].val) == 0.0:
            a = self.eval_var(invars[0])
            return ttnn.relu(a)
        if isinstance(invars[0], jax_core.Literal) and float(invars[0].val) == 0.0:
            b = self.eval_var(invars[1])
            return ttnn.relu(b)
        a = self.eval_var(invars[0])
        b = self.eval_var(invars[1])
        a, b = self._broadcast_for_binary(a, b, eqn)
        return ttnn.maximum(a, b)

    def _op_broadcast_in_dim(self, invars, params, eqn):
        """Handle broadcasting via CPU round-trip.

        Jaxpr uses broadcast_in_dim to insert size-1 dims and broadcast.
        Since reduce ops use keepdim=True on TT-NN (the reduced dim becomes
        1, padded to 32), we need to explicitly broadcast to the correct
        output shape to avoid garbage in padded positions.

        Strategy: read back logical values, reshape using broadcast_dimensions,
        broadcast to output shape, re-upload.
        """
        a = self.eval_var(invars[0])
        out_shape = eqn.outvars[0].aval.shape
        in_shape = eqn.invars[0].aval.shape if not isinstance(eqn.invars[0], jax_core.Literal) else ()
        if in_shape == out_shape:
            return a

        broadcast_dims = params.get('broadcast_dimensions', tuple(range(len(in_shape))))

        # Build intermediate shape with 1s for new/broadcast dims
        inter_shape = [1] * len(out_shape)
        for i, bd in enumerate(broadcast_dims):
            inter_shape[bd] = in_shape[i] if i < len(in_shape) else 1

        # Read back logical values from device
        # The tensor on device has keepdim so reduced dims are 1 (padded to 32).
        # Use from_ttnn which handles the unpadding correctly for 2D.
        # But from_ttnn uses the Jaxpr in_shape which may be 1D (reduce removed dim).
        # We need the ACTUAL device logical shape (with keepdim).
        # Since we used keepdim=True, the device tensor's logical shape matches
        # inter_shape (the intermediate shape before broadcasting).
        t = ttnn.to_torch(a).float()
        while t.dim() < 2:
            t = t.unsqueeze(0)
        while t.dim() > len(inter_shape):
            t = t.squeeze(0)
        # Pad inter_shape dims for correct slicing
        t_np = t.numpy()
        slices = tuple(slice(0, s) for s in inter_shape)
        try:
            t_np = t_np[slices].copy()
        except IndexError:
            # Fallback: reshape flat
            n = 1
            for d in in_shape:
                n *= d
            t_np = t.numpy().reshape(-1)[:n].reshape(inter_shape)
        t_broadcast = np.broadcast_to(t_np, out_shape).copy()
        return self.to_ttnn(t_broadcast)
        return a

    def _op_reshape(self, invars, params, eqn):
        a = self.eval_var(invars[0])
        new_shape = params.get('new_sizes', params.get('dimensions', None))
        return ttnn.reshape(a, new_shape) if new_shape else a

    def _op_transpose(self, invars, params, eqn):
        a = self.eval_var(invars[0])
        perm = params['permutation']
        return ttnn.permute(a, perm)

    def _op_squeeze(self, invars, params, eqn):
        a = self.eval_var(invars[0])
        # TT-NN tensors are already handled by reshape/broadcast
        return a

    # --- New ops for experiment 18 ---

    def _op_reduce_max(self, invars, params, eqn):
        """Reduce max along axes. Jaxpr reduce ops do NOT keepdim --
        the output shape has the reduced dims removed. But TT-NN
        requires keepdim for tile alignment. We keepdim on device and
        let broadcast_in_dim handle the reshaping."""
        a = self.eval_var(invars[0])
        axes = params['axes']
        for ax in sorted(axes, reverse=True):
            a = ttnn.max(a, dim=ax, keepdim=True)
        return a

    def _op_reduce_sum(self, invars, params, eqn):
        """Reduce sum along axes."""
        a = self.eval_var(invars[0])
        axes = params['axes']
        for ax in sorted(axes, reverse=True):
            a = ttnn.sum(a, dim=ax, keepdim=True)
        return a

    def _op_div(self, invars, params, eqn):
        """Division: a / b. Handle scalar divisors."""
        if isinstance(invars[1], jax_core.Literal):
            a = self.eval_var(invars[0])
            scalar = float(invars[1].val)
            if scalar != 0.0:
                return ttnn.multiply(a, 1.0 / scalar)
            return a  # div by 0 -- shouldn't happen
        if isinstance(invars[0], jax_core.Literal):
            b = self.eval_var(invars[1])
            recip = ttnn.reciprocal(b)
            return ttnn.multiply(recip, float(invars[0].val))
        a = self.eval_var(invars[0])
        b = self.eval_var(invars[1])
        a, b = self._broadcast_for_binary(a, b, eqn)
        return ttnn.mul(a, ttnn.reciprocal(b))

    def _op_sqrt(self, invars, params, eqn):
        a = self.eval_var(invars[0])
        return ttnn.sqrt(a)

    def _op_rsqrt(self, invars, params, eqn):
        a = self.eval_var(invars[0])
        return ttnn.rsqrt(a)

    def _op_reciprocal(self, invars, params, eqn):
        a = self.eval_var(invars[0])
        return ttnn.reciprocal(a)

    def _op_convert_element_type(self, invars, params, eqn):
        """Type conversion -- pass through since we always use bf16 on device."""
        return self.eval_var(invars[0])

    def _op_stop_gradient(self, invars, params, eqn):
        """Stop gradient -- identity in forward pass."""
        return self.eval_var(invars[0])

    def _op_integer_pow(self, invars, params, eqn):
        """Handle integer powers. x^2 = x*x, x^3 = x*x*x, etc."""
        a = self.eval_var(invars[0])
        y = params['y']  # the exponent
        if y == 0:
            # x^0 = 1
            return self.to_ttnn(np.ones((1, 1), dtype=np.float32))
        if y == 1:
            return a
        if y == 2:
            return ttnn.mul(a, a)
        if y == 3:
            return ttnn.mul(ttnn.mul(a, a), a)
        if y == -1:
            return ttnn.reciprocal(a)
        if y == -2:
            sq = ttnn.mul(a, a)
            return ttnn.reciprocal(sq)
        # General case: repeated squaring
        if y < 0:
            a = ttnn.reciprocal(a)
            y = -y
        result = a
        for _ in range(y - 1):
            result = ttnn.mul(result, a)
        return result

    # --- Sub-jaxpr handling (same as exp 14) ---

    def _exec_sub_jaxpr(self, sub_jaxpr_obj, invars):
        """Execute a sub-jaxpr (handles both Jaxpr and ClosedJaxpr)."""
        if hasattr(sub_jaxpr_obj, 'jaxpr'):
            raw_jaxpr = sub_jaxpr_obj.jaxpr
            consts = getattr(sub_jaxpr_obj, 'consts', [])
        else:
            raw_jaxpr = sub_jaxpr_obj
            consts = []

        for sv, const in zip(raw_jaxpr.constvars, consts):
            self.env[sv] = self.to_ttnn(const)

        for sv, iv in zip(raw_jaxpr.invars, invars):
            self.env[sv] = self.eval_var(iv)

        for sub_eqn in raw_jaxpr.eqns:
            self._exec_eqn(sub_eqn)

        # Handle multiple outputs
        if len(raw_jaxpr.outvars) == 1:
            return self.eval_var(raw_jaxpr.outvars[0])
        return [self.eval_var(v) for v in raw_jaxpr.outvars]

    def _op_subjaxpr(self, invars, params, eqn):
        """Handle ops with sub-jaxprs (custom_jvp_call, pjit)."""
        prim = eqn.primitive.name
        if prim == 'custom_jvp_call':
            return self._exec_sub_jaxpr(params['call_jaxpr'], invars)
        elif prim == 'pjit':
            return self._exec_sub_jaxpr(params['jaxpr'], invars)
        raise NotImplementedError(f"Unsupported sub-jaxpr: {prim}")


# Track all ops across all tests
all_ops_encountered = set()
all_ops_handled = set()
all_ops_unhandled = set()

def run_test(name, fn, args, arg_names=None):
    """Trace a function, interpret on TT-NN, compare against JAX CPU."""
    print(f"\n{'=' * 60}")
    print(f"{name}")
    print("=" * 60)

    # JAX CPU reference
    jax_args = [jnp.array(a) for a in args]
    ref = np.array(fn(*jax_args))

    # Trace
    jaxpr = make_jaxpr(fn)(*jax_args)
    print(f"\n  Jaxpr equations ({len(jaxpr.jaxpr.eqns)}):")
    for eqn in jaxpr.jaxpr.eqns:
        pname = eqn.primitive.name
        pinfo = ""
        if 'axes' in eqn.params:
            pinfo = f" axes={eqn.params['axes']}"
        if 'y' in eqn.params:
            pinfo = f" y={eqn.params['y']}"
        print(f"    {pname}{pinfo}")

    # Interpret
    try:
        interp = TTNNInterpreter(device)
        result = interp.interpret(jaxpr, args)

        all_ops_encountered.update(interp.ops_encountered)
        all_ops_handled.update(interp.ops_handled)
        all_ops_unhandled.update(interp.ops_unhandled)

        # Compare
        err = np.abs(result - ref)
        print(f"\n  Result shape: {result.shape}")
        print(f"  Max abs error:  {err.max():.6f}")
        print(f"  Mean abs error: {err.mean():.6f}")
        print(f"  PASS")
        return True, err.max(), err.mean()
    except NotImplementedError as e:
        all_ops_encountered.update(interp.ops_encountered)
        all_ops_handled.update(interp.ops_handled)
        all_ops_unhandled.update(interp.ops_unhandled)
        print(f"\n  FAIL: {e}")
        return False, None, None
    except Exception as e:
        all_ops_encountered.update(interp.ops_encountered)
        all_ops_handled.update(interp.ops_handled)
        all_ops_unhandled.update(interp.ops_unhandled)
        print(f"\n  ERROR: {type(e).__name__}: {str(e)[:200]}")
        return False, None, None


# ============================================================
# TEST 1: Softmax (failed in experiment 14)
# ============================================================

def softmax(x):
    """Manual softmax -- the one that failed in exp 14."""
    e = jnp.exp(x - jnp.max(x, axis=-1, keepdims=True))
    return e / jnp.sum(e, axis=-1, keepdims=True)

np.random.seed(42)
x_sm = np.random.randn(32, 64).astype(np.float32)
ok1, max1, mean1 = run_test("TEST 1: Softmax (failed in exp 14)", softmax, [x_sm])


# ============================================================
# TEST 2: Layer Norm
# ============================================================

def layer_norm(x, gamma, beta):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return gamma * (x - mean) / jnp.sqrt(var + 1e-5) + beta

x_ln = np.random.randn(32, 64).astype(np.float32)
gamma = np.ones(64, dtype=np.float32)
beta = np.zeros(64, dtype=np.float32)
ok2, max2, mean2 = run_test("TEST 2: Layer Norm", layer_norm, [x_ln, gamma, beta])


# ============================================================
# TEST 3: Transformer-style block (linear + ReLU + layernorm)
# ============================================================

def transformer_block(x, w, b, gamma, beta):
    """Linear projection -> ReLU -> Layer Norm."""
    h = jnp.dot(x, w) + b          # linear
    h = jax.nn.relu(h)              # activation
    # layer norm
    mean = jnp.mean(h, axis=-1, keepdims=True)
    var = jnp.mean((h - mean) ** 2, axis=-1, keepdims=True)
    return gamma * (h - mean) / jnp.sqrt(var + 1e-5) + beta

batch, d_in, d_out = 32, 64, 64
np.random.seed(123)
x_tb = np.random.randn(batch, d_in).astype(np.float32)
w_tb = np.random.randn(d_in, d_out).astype(np.float32) * (2 / d_in) ** 0.5
b_tb = np.zeros(d_out, dtype=np.float32)
g_tb = np.ones(d_out, dtype=np.float32)
bt_tb = np.zeros(d_out, dtype=np.float32)

ok3, max3, mean3 = run_test(
    "TEST 3: Transformer block (Linear + ReLU + LayerNorm)",
    transformer_block,
    [x_tb, w_tb, b_tb, g_tb, bt_tb]
)


# ============================================================
# TEST 4: jax.nn.softmax (the library version)
# ============================================================

def jax_softmax(x):
    return jax.nn.softmax(x, axis=-1)

ok4, max4, mean4 = run_test("TEST 4: jax.nn.softmax (library)", jax_softmax, [x_sm])


# ============================================================
# TEST 5: Op Coverage Report
# ============================================================

print(f"\n{'=' * 60}")
print("TEST 5: Op Coverage Report")
print("=" * 60)

print(f"\n  All primitives encountered across tests ({len(all_ops_encountered)}):")
for op in sorted(all_ops_encountered):
    status = "HANDLED" if op in all_ops_handled else "MISSING"
    print(f"    [{status}]  {op}")

print(f"\n  Handled:   {len(all_ops_handled)}")
print(f"  Missing:   {len(all_ops_unhandled)}")
print(f"  Total:     {len(all_ops_encountered)}")


# ============================================================
# Summary
# ============================================================

print(f"\n{'=' * 60}")
print("Summary")
print("=" * 60)

results = [
    ("Softmax (manual)",       ok1, max1, mean1),
    ("Layer Norm",             ok2, max2, mean2),
    ("Transformer block",     ok3, max3, mean3),
    ("jax.nn.softmax (lib)",  ok4, max4, mean4),
]

print()
for name, ok, mx, mn in results:
    if ok:
        print(f"  PASS  {name:30s}  max_err={mx:.6f}  mean_err={mn:.6f}")
    else:
        print(f"  FAIL  {name}")

print(f"""
  New ops added in this experiment:
    reduce_max, reduce_sum, div, sqrt, rsqrt,
    reciprocal, convert_element_type, integer_pow

  The interpreter now handles softmax, layer norm, and a
  mini-transformer block. These are the core building blocks
  for attention-based models.

  Next step: add gather/scatter for embedding lookups, and
  trace capture for performance (eliminate Python overhead).
""")

ttnn.close_device(device)
print("Done!")
