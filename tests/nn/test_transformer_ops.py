# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the Transformer-era functional primitives.

Each op is pinned against an explicit NumPy reference so a future change to the
composition (or to any primitive underneath it) is caught here rather than
silently altering model behavior.
"""

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F

EPS_FD = 1e-6


def _t(array, requires_grad=False):
    return mt.Tensor(
        np.asarray(array, dtype="float64").tolist(),
        dtype="float64",
        requires_grad=requires_grad,
    )


def _finite_diff(fn, x, grad_output):
    """Central-difference gradient of sum(fn(x) * grad_output) wrt x."""
    num = np.zeros_like(x)
    for idx in np.ndindex(*x.shape):
        plus, minus = x.copy(), x.copy()
        plus[idx] += EPS_FD
        minus[idx] -= EPS_FD
        num[idx] = (
            np.sum(fn(plus) * grad_output) - np.sum(fn(minus) * grad_output)
        ) / (2 * EPS_FD)
    return num


# --------------------------------------------------------------------------
# RMSNorm
# --------------------------------------------------------------------------


def _rms_norm_np(x, weight, eps):
    return x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps) * weight


def test_rms_norm_matches_reference():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 8))
    w = rng.standard_normal(8)
    eps = 1e-6

    out = F.rms_norm(_t(x), 8, _t(w), eps)

    np.testing.assert_allclose(out.numpy(), _rms_norm_np(x, w, eps), rtol=1e-12)


def test_rms_norm_has_no_mean_subtraction():
    # A constant row has zero variance; LayerNorm would map it to 0, but
    # RMSNorm divides by the RMS and leaves the sign/scale information.
    out = F.rms_norm(_t([[3.0, 3.0, 3.0, 3.0]]), 4, None, 0.0)
    np.testing.assert_allclose(out.numpy(), [[1.0, 1.0, 1.0, 1.0]], atol=1e-9)


def test_rms_norm_gradients_match_finite_differences():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 4))
    w = rng.standard_normal(4)
    g = rng.standard_normal((3, 4))
    eps = 1e-6

    xt, wt = _t(x, requires_grad=True), _t(w, requires_grad=True)
    (F.rms_norm(xt, 4, wt, eps) * _t(g)).sum().backward()

    np.testing.assert_allclose(
        xt.grad.numpy(),
        _finite_diff(lambda v: _rms_norm_np(v, w, eps), x, g),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        wt.grad.numpy(),
        _finite_diff(lambda v: _rms_norm_np(x, v, eps), w, g),
        atol=1e-6,
    )


# --------------------------------------------------------------------------
# Scaled dot-product attention
# --------------------------------------------------------------------------


def _sdpa_np(q, k, v, mask=None, causal=False, scale=None):
    scale = 1.0 / np.sqrt(q.shape[-1]) if scale is None else scale
    s = (q @ np.swapaxes(k, -1, -2)) * scale
    if mask is not None:
        s = s + mask
    if causal:
        length, keys = s.shape[-2], s.shape[-1]
        i = np.arange(length)[:, None]
        j = np.arange(keys)[None, :]
        s = np.where((j - i) > (keys - length), -np.inf, s)
    s = s - s.max(axis=-1, keepdims=True)
    p = np.exp(s)
    return (p / p.sum(axis=-1, keepdims=True)) @ v


@pytest.fixture
def qkv():
    rng = np.random.default_rng(2)
    return (
        rng.standard_normal((2, 3, 5, 4)),
        rng.standard_normal((2, 3, 5, 4)),
        rng.standard_normal((2, 3, 5, 6)),
    )


def test_sdpa_matches_reference_with_broadcast_batch_dims(qkv):
    q, k, v = qkv
    out = F.scaled_dot_product_attention(_t(q), _t(k), _t(v))
    # Leading (batch, heads) axes must ride through untouched.
    assert tuple(out.shape) == (2, 3, 5, 6)
    np.testing.assert_allclose(out.numpy(), _sdpa_np(q, k, v), rtol=1e-12)


def test_sdpa_causal_matches_reference(qkv):
    q, k, v = qkv
    out = F.scaled_dot_product_attention(_t(q), _t(k), _t(v), is_causal=True)
    np.testing.assert_allclose(out.numpy(), _sdpa_np(q, k, v, causal=True), rtol=1e-12)


def test_sdpa_causal_ignores_future_positions(qkv):
    q, k, v = qkv
    baseline = F.scaled_dot_product_attention(
        _t(q), _t(k), _t(v), is_causal=True
    ).numpy()

    # Perturbing the last key/value position must not reach query position 0.
    k2, v2 = k.copy(), v.copy()
    k2[..., -1, :] += 10.0
    v2[..., -1, :] += 10.0
    perturbed = F.scaled_dot_product_attention(
        _t(q), _t(k2), _t(v2), is_causal=True
    ).numpy()

    np.testing.assert_allclose(perturbed[..., 0, :], baseline[..., 0, :], rtol=1e-12)
    assert np.abs(perturbed[..., -1, :] - baseline[..., -1, :]).max() > 1e-6


def test_sdpa_float_mask_is_additive(qkv):
    q, k, v = qkv
    rng = np.random.default_rng(3)
    mask = rng.standard_normal((5, 5))
    out = F.scaled_dot_product_attention(_t(q), _t(k), _t(v), attn_mask=_t(mask))
    np.testing.assert_allclose(out.numpy(), _sdpa_np(q, k, v, mask=mask), rtol=1e-12)


def test_sdpa_custom_scale(qkv):
    q, k, v = qkv
    out = F.scaled_dot_product_attention(_t(q), _t(k), _t(v), scale=0.5)
    np.testing.assert_allclose(out.numpy(), _sdpa_np(q, k, v, scale=0.5), rtol=1e-12)


def test_sdpa_rejects_mask_together_with_causal(qkv):
    q, k, v = qkv
    with pytest.raises(Exception):
        F.scaled_dot_product_attention(
            _t(q), _t(k), _t(v), attn_mask=_t(np.zeros((5, 5))), is_causal=True
        )


def test_sdpa_gradients_match_finite_differences():
    rng = np.random.default_rng(4)
    q = rng.standard_normal((1, 2, 3))
    k = rng.standard_normal((1, 2, 3))
    v = rng.standard_normal((1, 2, 3))
    g = rng.standard_normal((1, 2, 3))

    qt, kt, vt = (_t(a, requires_grad=True) for a in (q, k, v))
    (F.scaled_dot_product_attention(qt, kt, vt) * _t(g)).sum().backward()

    np.testing.assert_allclose(
        qt.grad.numpy(), _finite_diff(lambda a: _sdpa_np(a, k, v), q, g), atol=1e-6
    )
    np.testing.assert_allclose(
        kt.grad.numpy(), _finite_diff(lambda a: _sdpa_np(q, a, v), k, g), atol=1e-6
    )
    np.testing.assert_allclose(
        vt.grad.numpy(), _finite_diff(lambda a: _sdpa_np(q, k, a), v, g), atol=1e-6
    )


# --------------------------------------------------------------------------
# Rotary position embedding
# --------------------------------------------------------------------------


def _rope_np(x, base=10000.0, offset=0):
    seq, dim = x.shape[-2], x.shape[-1]
    half = dim // 2
    inv_freq = base ** (-(np.arange(0, half) * 2.0) / dim)
    freqs = np.outer(np.arange(offset, offset + seq), inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    rotated = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * np.cos(emb) + rotated * np.sin(emb)


@pytest.mark.parametrize("base,offset", [(10000.0, 0), (10000.0, 3), (5000.0, 0)])
def test_rope_matches_reference(base, offset):
    rng = np.random.default_rng(5)
    x = rng.standard_normal((2, 3, 5, 8))
    out = F.rope(_t(x), base=base, offset=offset)
    np.testing.assert_allclose(out.numpy(), _rope_np(x, base, offset), rtol=1e-12)


def test_rope_is_a_rotation_so_it_preserves_norm():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((2, 4, 6))
    out = F.rope(_t(x)).numpy()
    np.testing.assert_allclose(
        np.linalg.norm(out, axis=-1), np.linalg.norm(x, axis=-1), rtol=1e-12
    )


def test_rope_is_identity_at_position_zero():
    # Position 0 has all angles zero, so cos=1 and sin=0.
    x = [[1.0, 2.0, 3.0, 4.0]]
    np.testing.assert_allclose(F.rope(_t(x)).numpy(), x, atol=1e-12)


def test_rope_offset_shifts_positions():
    # Row p at offset o must equal row p+o computed from position 0.
    rng = np.random.default_rng(7)
    x = rng.standard_normal((1, 4))
    shifted = F.rope(_t(x), offset=2).numpy()
    long = _rope_np(np.repeat(x, 3, axis=0))
    np.testing.assert_allclose(shifted[0], long[2], rtol=1e-12)


def test_rope_rejects_odd_head_dim():
    with pytest.raises(Exception):
        F.rope(_t(np.zeros((2, 3))))


def test_rope_gradients_match_finite_differences():
    rng = np.random.default_rng(8)
    x = rng.standard_normal((3, 4))
    g = rng.standard_normal((3, 4))
    xt = _t(x, requires_grad=True)
    (F.rope(xt) * _t(g)).sum().backward()
    np.testing.assert_allclose(xt.grad.numpy(), _finite_diff(_rope_np, x, g), atol=1e-6)


# --------------------------------------------------------------------------
# Gated linear unit
# --------------------------------------------------------------------------


def _glu_np(x, axis=-1):
    a, b = np.split(x, 2, axis=axis)
    return a * (1.0 / (1.0 + np.exp(-b)))


@pytest.mark.parametrize("shape,dim", [((4, 6), -1), ((6, 4), 0), ((2, 4, 6), 1)])
def test_glu_matches_reference(shape, dim):
    rng = np.random.default_rng(9)
    x = rng.standard_normal(shape)
    out = F.glu(_t(x), dim=dim)
    np.testing.assert_allclose(out.numpy(), _glu_np(x, dim), rtol=1e-12)


def test_glu_halves_the_split_dimension():
    out = F.glu(_t(np.zeros((3, 8))))
    assert tuple(out.shape) == (3, 4)


def test_glu_rejects_odd_split_dimension():
    with pytest.raises(Exception):
        F.glu(_t(np.zeros((3, 5))))


def test_glu_gradients_match_finite_differences():
    rng = np.random.default_rng(10)
    x = rng.standard_normal((4, 6))
    g = rng.standard_normal((4, 3))
    xt = _t(x, requires_grad=True)
    (F.glu(xt) * _t(g)).sum().backward()
    np.testing.assert_allclose(xt.grad.numpy(), _finite_diff(_glu_np, x, g), atol=1e-6)
