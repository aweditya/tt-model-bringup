"""Test basic JAX operations through the full PJRT pipeline (Phase 3).

These tests verify that jax.jit(f)(x) works end-to-end on the TT backend:
JAX → StableHLO → PJRT Compile → PJRT Execute → result.
"""

import os
import pytest
import numpy as np

# In device mode (TT_PJRT_USE_DEVICE=1) the engine runs ops in bf16 on the
# Blackhole. Numpy mode runs ops in fp32 on CPU. Tolerances must adapt:
# bf16 has ~7 bits of mantissa, so transcendentals/matmuls/reductions visibly
# drift by 1e-3 to 1e-2. We use one envelope per mode.
_DEVICE_MODE = os.environ.get('TT_PJRT_USE_DEVICE', '0') == '1'
_ATOL = 5e-2 if _DEVICE_MODE else 1e-5
_RTOL = 5e-2 if _DEVICE_MODE else 1e-4


def assert_close(actual, expected, atol=None, rtol=None):
    """assert_close with mode-adapted defaults.

    In device mode, explicit tolerances from the test are IGNORED and
    replaced with the bf16 envelope — bf16 cannot meet fp32-style
    tolerances no matter how the test is written.
    """
    if _DEVICE_MODE:
        # bf16 floor: 5%. Accept any explicit tolerance from the test that's
        # MORE permissive (e.g., a deep matmul that needs atol=1.0).
        effective_atol = max(_ATOL, atol or 0.0)
        effective_rtol = max(_RTOL, rtol or 0.0)
        np.testing.assert_allclose(actual, expected,
                                    atol=effective_atol, rtol=effective_rtol)
    else:
        np.testing.assert_allclose(
            actual, expected,
            atol=_ATOL if atol is None else atol,
            rtol=_RTOL if rtol is None else rtol,
        )


class TestArithmetic:
    def test_add_scalar(self, tt_device):
        """jax.jit(lambda x: x + 1)(x)"""
        import jax
        f = jax.jit(lambda x: x + 1.0)
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = f(jax.device_put(x, tt_device))
        result = jax.device_get(result)
        assert_close(result, x + 1.0, atol=1e-6)

    def test_multiply_add(self, tt_device):
        """jax.jit(lambda x: x * 2 + 3)(x)"""
        import jax
        f = jax.jit(lambda x: x * 2.0 + 3.0)
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = jax.device_get(f(jax.device_put(x, tt_device)))
        assert_close(result, x * 2.0 + 3.0, atol=1e-6)

    def test_subtract(self, tt_device):
        """jax.jit(lambda x, y: x - y)(x, y)"""
        import jax
        f = jax.jit(lambda x, y: x - y)
        x = np.array([5.0, 4.0, 3.0, 2.0], dtype=np.float32)
        y = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        result = jax.device_get(f(
            jax.device_put(x, tt_device),
            jax.device_put(y, tt_device),
        ))
        assert_close(result, x - y, atol=1e-6)

    def test_divide(self, tt_device):
        """jax.jit(lambda x, y: x / y)(x, y)"""
        import jax
        f = jax.jit(lambda x, y: x / y)
        x = np.array([6.0, 8.0, 9.0, 12.0], dtype=np.float32)
        y = np.array([2.0, 4.0, 3.0, 6.0], dtype=np.float32)
        result = jax.device_get(f(
            jax.device_put(x, tt_device),
            jax.device_put(y, tt_device),
        ))
        assert_close(result, x / y, atol=1e-6)


class TestUnaryOps:
    def test_negate(self, tt_device):
        import jax
        f = jax.jit(lambda x: -x)
        x = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
        result = jax.device_get(f(jax.device_put(x, tt_device)))
        assert_close(result, -x, atol=1e-6)

    def test_exp(self, tt_device):
        import jax
        import jax.numpy as jnp
        f = jax.jit(lambda x: jnp.exp(x))
        x = np.array([0.0, 1.0, 2.0, -1.0], dtype=np.float32)
        result = jax.device_get(f(jax.device_put(x, tt_device)))
        assert_close(result, np.exp(x), rtol=1e-5)

    def test_tanh(self, tt_device):
        import jax
        import jax.numpy as jnp
        f = jax.jit(lambda x: jnp.tanh(x))
        x = np.array([0.0, 1.0, -1.0, 3.0], dtype=np.float32)
        result = jax.device_get(f(jax.device_put(x, tt_device)))
        assert_close(result, np.tanh(x), rtol=1e-5)

    def test_relu(self, tt_device):
        """max(x, 0)"""
        import jax
        import jax.numpy as jnp
        f = jax.jit(lambda x: jnp.maximum(x, 0.0))
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        result = jax.device_get(f(jax.device_put(x, tt_device)))
        assert_close(result, np.maximum(x, 0.0), atol=1e-6)


class TestMatmul:
    def test_simple_matmul(self, tt_device):
        """x @ w"""
        import jax
        f = jax.jit(lambda x, w: x @ w)
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        w = np.eye(3, 4, dtype=np.float32) * 2
        result = jax.device_get(f(
            jax.device_put(x, tt_device),
            jax.device_put(w, tt_device),
        ))
        assert_close(result, x @ w, atol=1e-5)

    def test_linear_layer(self, tt_device):
        """x @ w + b"""
        import jax
        f = jax.jit(lambda x, w, b: x @ w + b)
        x = np.random.randn(2, 3).astype(np.float32)
        w = np.random.randn(3, 4).astype(np.float32)
        b = np.random.randn(4).astype(np.float32)
        result = jax.device_get(f(
            jax.device_put(x, tt_device),
            jax.device_put(w, tt_device),
            jax.device_put(b, tt_device),
        ))
        assert_close(result, x @ w + b, rtol=1e-5)

    def test_larger_matmul(self, tt_device):
        """64x128 @ 128x32"""
        import jax
        f = jax.jit(lambda x, w: x @ w)
        x = np.random.randn(64, 128).astype(np.float32)
        w = np.random.randn(128, 32).astype(np.float32)
        result = jax.device_get(f(
            jax.device_put(x, tt_device),
            jax.device_put(w, tt_device),
        ))
        # 128-deep matmul: bf16 drifts by ~5% on K=128, with one or two
        # near-zero outliers further out. Allow a generous absolute floor.
        assert_close(result, x @ w, atol=1.0, rtol=1e-4)


class TestReduce:
    def test_sum(self, tt_device):
        """sum(x, axis=-1) through full PJRT pipeline"""
        import jax
        import jax.numpy as jnp
        f = jax.jit(lambda x: jnp.sum(x, axis=-1))
        x = np.random.randn(4, 8).astype(np.float32)
        result = jax.device_get(f(jax.device_put(x, tt_device)))
        assert_close(result, np.sum(x, axis=-1), rtol=1e-5)

    def test_max(self, tt_device):
        """max(x, axis=-1) through full PJRT pipeline"""
        import jax
        import jax.numpy as jnp
        f = jax.jit(lambda x: jnp.max(x, axis=-1))
        x = np.random.randn(4, 8).astype(np.float32)
        result = jax.device_get(f(jax.device_put(x, tt_device)))
        assert_close(result, np.max(x, axis=-1))


class TestComposite:
    def test_softmax(self, tt_device):
        """jax.nn.softmax through full PJRT pipeline"""
        import jax
        f = jax.jit(lambda x: jax.nn.softmax(x, axis=-1))
        x = np.random.randn(2, 64).astype(np.float32)
        result = jax.device_get(f(jax.device_put(x, tt_device)))
        ref = np.exp(x - np.max(x, axis=-1, keepdims=True))
        ref = ref / np.sum(ref, axis=-1, keepdims=True)
        assert_close(result, ref, rtol=1e-5)

    def test_layer_norm(self, tt_device):
        """Manual layer norm through full PJRT pipeline"""
        import jax
        import jax.numpy as jnp

        @jax.jit
        def layer_norm(x, g, b):
            mean = jnp.mean(x, axis=-1, keepdims=True)
            var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
            return g * (x - mean) / jnp.sqrt(var + 1e-5) + b

        x = np.random.randn(2, 64).astype(np.float32)
        g = np.ones(64, dtype=np.float32)
        b = np.zeros(64, dtype=np.float32)
        result = jax.device_get(layer_norm(
            jax.device_put(x, tt_device),
            jax.device_put(g, tt_device),
            jax.device_put(b, tt_device),
        ))
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
        ref = g * (x - mean) / np.sqrt(var + 1e-5) + b
        assert_close(result, ref, rtol=1e-5)

    def test_rms_norm(self, tt_device):
        """RMS norm through full PJRT pipeline"""
        import jax
        import jax.numpy as jnp

        @jax.jit
        def rms_norm(x, g):
            ms = jnp.mean(x ** 2, axis=-1, keepdims=True)
            return g * x / jnp.sqrt(ms + 1e-6)

        x = np.random.randn(2, 64).astype(np.float32)
        g = np.ones(64, dtype=np.float32)
        result = jax.device_get(rms_norm(
            jax.device_put(x, tt_device),
            jax.device_put(g, tt_device),
        ))
        ms = np.mean(x ** 2, axis=-1, keepdims=True)
        ref = g * x / np.sqrt(ms + 1e-6)
        assert_close(result, ref, rtol=1e-5)

    def test_mlp_with_relu(self, tt_device):
        """MLP with relu through full PJRT pipeline"""
        import jax

        @jax.jit
        def mlp(x, w1, b1, w2, b2):
            h = jax.nn.relu(x @ w1 + b1)
            return h @ w2 + b2

        np.random.seed(42)
        x = np.random.randn(2, 32).astype(np.float32) * 0.1
        w1 = np.random.randn(32, 64).astype(np.float32) * 0.1
        b1 = np.zeros(64, dtype=np.float32)
        w2 = np.random.randn(64, 32).astype(np.float32) * 0.1
        b2 = np.zeros(32, dtype=np.float32)
        result = jax.device_get(mlp(
            jax.device_put(x, tt_device),
            jax.device_put(w1, tt_device),
            jax.device_put(b1, tt_device),
            jax.device_put(w2, tt_device),
            jax.device_put(b2, tt_device),
        ))
        ref = np.maximum(x @ w1 + b1, 0) @ w2 + b2
        assert_close(result, ref, rtol=1e-4, atol=1e-5)

    def test_attention(self, tt_device):
        """Single-head self-attention through full PJRT pipeline"""
        import jax
        import jax.numpy as jnp

        @jax.jit
        def attention(x, wq, wk, wv, wo):
            q = x @ wq
            k = x @ wk
            v = x @ wv
            d = jnp.float32(q.shape[-1])
            scores = jax.nn.softmax(q @ k.T / jnp.sqrt(d), axis=-1)
            return (scores @ v) @ wo

        D = 32
        np.random.seed(42)
        x = np.random.randn(8, D).astype(np.float32) * 0.1
        wq = np.random.randn(D, D).astype(np.float32) * 0.1
        wk = np.random.randn(D, D).astype(np.float32) * 0.1
        wv = np.random.randn(D, D).astype(np.float32) * 0.1
        wo = np.random.randn(D, D).astype(np.float32) * 0.1
        result = jax.device_get(attention(
            jax.device_put(x, tt_device),
            jax.device_put(wq, tt_device),
            jax.device_put(wk, tt_device),
            jax.device_put(wv, tt_device),
            jax.device_put(wo, tt_device),
        ))
        q = x @ wq; k = x @ wk; v = x @ wv
        scores = q @ k.T / np.sqrt(D)
        scores_exp = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)
        ref = (attn @ v) @ wo
        assert_close(result, ref, rtol=1e-4, atol=1e-5)


class TestSlice:
    def test_static_slice(self, tt_device):
        """Slice through full PJRT pipeline"""
        import jax
        f = jax.jit(lambda x: x[:, :, :8, :])
        x = np.random.randn(1, 4, 32, 16).astype(np.float32)
        result = jax.device_get(f(jax.device_put(x, tt_device)))
        assert_close(result, x[:, :, :8, :])


class TestCompareSelect:
    def test_where(self, tt_device):
        """jnp.where through full PJRT pipeline"""
        import jax
        import jax.numpy as jnp
        f = jax.jit(lambda x: jnp.where(x > 0, x, 0.0))
        x = np.array([-2, -1, 0, 0.5, 1, 2, -0.5, 3], dtype=np.float32)
        result = jax.device_get(f(jax.device_put(x, tt_device)))
        assert_close(result, np.where(x > 0, x, 0.0))

    def test_tril(self, tt_device):
        """jnp.tril (iota + compare + select) through full PJRT pipeline"""
        import jax
        import jax.numpy as jnp
        f = jax.jit(lambda x: jnp.tril(x))
        x = np.ones((8, 8), dtype=np.float32)
        result = jax.device_get(f(jax.device_put(x, tt_device)))
        assert_close(result, np.tril(x))


class TestConcatenate:
    def test_concat(self, tt_device):
        """Concatenate through full PJRT pipeline"""
        import jax
        import jax.numpy as jnp
        f = jax.jit(lambda x, y: jnp.concatenate([x, y], axis=-1))
        x = np.random.randn(2, 3).astype(np.float32)
        y = np.random.randn(2, 4).astype(np.float32)
        result = jax.device_get(f(
            jax.device_put(x, tt_device),
            jax.device_put(y, tt_device),
        ))
        assert_close(result, np.concatenate([x, y], axis=-1))


class TestScatterGather:
    def test_kv_cache_update(self, tt_device):
        """scatter (KV cache .at[].set()) through full PJRT pipeline"""
        import jax
        import jax.numpy as jnp

        @jax.jit
        def kv_update(cache, new_kv):
            return cache.at[:, :, 5:6, :].set(new_kv)

        cache = np.zeros((1, 4, 32, 16), dtype=np.float32)
        new_kv = np.ones((1, 4, 1, 16), dtype=np.float32) * 42.0
        result = jax.device_get(kv_update(
            jax.device_put(cache, tt_device),
            jax.device_put(new_kv, tt_device),
        ))
        expected = cache.copy()
        expected[:, :, 5:6, :] = new_kv
        assert_close(result, expected)

    def test_embedding_lookup(self, tt_device):
        """gather (table[ids]) through full PJRT pipeline"""
        import jax

        @jax.jit
        def embed(table, ids):
            return table[ids]

        table = np.random.randn(100, 64).astype(np.float32)
        ids = np.array([0, 5, 99], dtype=np.int32)
        result = jax.device_get(embed(
            jax.device_put(table, tt_device),
            jax.device_put(ids, tt_device),
        ))
        assert_close(result, table[ids])


class TestArgmax:
    def test_argmax(self, tt_device):
        """argmax (greedy decoding) through full PJRT pipeline"""
        import jax
        import jax.numpy as jnp

        @jax.jit
        def greedy(logits):
            return jnp.argmax(logits, axis=-1)

        x = np.random.randn(1, 100).astype(np.float32)
        result = jax.device_get(greedy(jax.device_put(x, tt_device)))
        np.testing.assert_array_equal(result, np.argmax(x, axis=-1))


class TestMultiOutput:
    def test_two_outputs(self, tt_device):
        """Function returning two values through full PJRT pipeline"""
        import jax

        @jax.jit
        def two_out(x):
            return x + 1.0, x * 2.0

        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        a, b = jax.device_get(two_out(jax.device_put(x, tt_device)))
        assert_close(a, x + 1.0)
        assert_close(b, x * 2.0)


class TestMHA:
    def test_multi_head_attention(self, tt_device):
        """Multi-head attention with reshape/transpose through PJRT"""
        import jax
        import jax.numpy as jnp

        @jax.jit
        def mha(x, wq, wk, wv, wo):
            q = (x @ wq).reshape(1, 8, 4, 16).transpose(0, 2, 1, 3)
            k = (x @ wk).reshape(1, 8, 4, 16).transpose(0, 2, 1, 3)
            v = (x @ wv).reshape(1, 8, 4, 16).transpose(0, 2, 1, 3)
            scores = jax.nn.softmax(
                q @ k.transpose(0, 1, 3, 2) / jnp.sqrt(16.0),
                axis=-1,
            )
            attn = (scores @ v).transpose(0, 2, 1, 3).reshape(1, 8, 64)
            return attn @ wo

        np.random.seed(42)
        x = np.random.randn(1, 8, 64).astype(np.float32) * 0.1
        wq = np.random.randn(64, 64).astype(np.float32) * 0.1
        wk = np.random.randn(64, 64).astype(np.float32) * 0.1
        wv = np.random.randn(64, 64).astype(np.float32) * 0.1
        wo = np.random.randn(64, 64).astype(np.float32) * 0.1
        result = jax.device_get(mha(
            jax.device_put(x, tt_device),
            jax.device_put(wq, tt_device),
            jax.device_put(wk, tt_device),
            jax.device_put(wv, tt_device),
            jax.device_put(wo, tt_device),
        ))
        # Numpy reference
        q = (x @ wq).reshape(1, 8, 4, 16).transpose(0, 2, 1, 3)
        k = (x @ wk).reshape(1, 8, 4, 16).transpose(0, 2, 1, 3)
        v = (x @ wv).reshape(1, 8, 4, 16).transpose(0, 2, 1, 3)
        scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(16.0)
        scores_exp = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)
        ref = (attn @ v).transpose(0, 2, 1, 3).reshape(1, 8, 64) @ wo
        assert_close(result, ref, rtol=1e-4, atol=1e-5)
