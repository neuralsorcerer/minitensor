# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the Transformer-era stateful layers.

Covers Embedding, LayerNorm, RMSNorm and MultiheadAttention: forward values
against NumPy references, parameter bookkeeping, the error paths, and that
gradients actually reach every learned tensor.
"""

import numpy as np
import pytest

import minitensor as mt
from minitensor import nn


def _t(array, requires_grad=False):
    return mt.Tensor(
        np.asarray(array, dtype="float64").tolist(),
        dtype="float64",
        requires_grad=requires_grad,
    )


def _ids(array):
    return mt.Tensor(np.asarray(array, dtype="int64").tolist(), dtype="int64")


# --------------------------------------------------------------------------
# nn.Embedding
# --------------------------------------------------------------------------


def test_embedding_gathers_rows_and_appends_embedding_dim():
    emb = nn.Embedding(10, 4, dtype="float64")
    weight = emb.weight.numpy()

    out = emb(_ids([[1, 3], [3, 7]]))

    assert tuple(out.shape) == (2, 2, 4)
    np.testing.assert_allclose(out.numpy(), weight[np.array([[1, 3], [3, 7]])])


def test_embedding_exposes_weight_as_its_only_parameter():
    emb = nn.Embedding(10, 4, dtype="float64")
    assert emb.num_embeddings == 10
    assert emb.embedding_dim == 4
    assert emb.padding_idx is None
    assert len(emb.parameters()) == 1
    assert emb.num_parameters() == 40


def test_embedding_padding_idx_produces_zero_rows():
    emb = nn.Embedding(6, 3, padding_idx=0, dtype="float64")
    out = emb(_ids([0, 2])).numpy()
    assert emb.padding_idx == 0
    np.testing.assert_allclose(out[0], np.zeros(3))


def test_embedding_padding_idx_blocks_gradient_to_that_row():
    emb = nn.Embedding(4, 2, padding_idx=1, dtype="float64")
    emb(_ids([1, 1, 3])).sum().backward()
    grad = emb.parameters()[0].grad.numpy()
    # Padded token contributes nothing, so its row keeps a zero gradient.
    np.testing.assert_allclose(grad[1], np.zeros(2))
    assert np.abs(grad[3]).sum() > 0


def test_embedding_accumulates_gradient_for_repeated_tokens():
    emb = nn.Embedding(3, 2, dtype="float64")
    emb(_ids([1, 1])).sum().backward()
    grad = emb.parameters()[0].grad.numpy()
    # Row 1 selected twice with an all-ones upstream gradient.
    np.testing.assert_allclose(grad[1], [2.0, 2.0])
    np.testing.assert_allclose(grad[0], [0.0, 0.0])
    np.testing.assert_allclose(grad[2], [0.0, 0.0])


@pytest.mark.parametrize("bad", [[99], [-1]])
def test_embedding_rejects_out_of_range_ids(bad):
    emb = nn.Embedding(5, 3, dtype="float64")
    with pytest.raises(Exception):
        emb(_ids(bad))


def test_embedding_rejects_float_indices():
    emb = nn.Embedding(5, 3, dtype="float64")
    with pytest.raises(Exception):
        emb(_t([0.0, 1.0]))


def test_embedding_rejects_padding_idx_outside_vocabulary():
    with pytest.raises(Exception):
        nn.Embedding(3, 2, padding_idx=3, dtype="float64")


# --------------------------------------------------------------------------
# nn.LayerNorm / nn.RMSNorm
# --------------------------------------------------------------------------


def _layer_norm_np(x, eps):
    mu = x.mean(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(x.var(axis=-1, keepdims=True) + eps)


def _rms_norm_np(x, eps):
    return x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)


def test_layer_norm_matches_reference():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 5))
    layer = nn.LayerNorm(5, dtype="float64")
    np.testing.assert_allclose(
        layer(_t(x)).numpy(), _layer_norm_np(x, layer.eps), rtol=1e-10
    )


def test_layer_norm_accepts_multi_dimensional_normalized_shape():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 2, 4))
    layer = nn.LayerNorm([2, 4], dtype="float64")
    assert layer.normalized_shape == [2, 4]

    flat = x.reshape(3, -1)
    expected = _layer_norm_np(flat, layer.eps).reshape(3, 2, 4)
    np.testing.assert_allclose(layer(_t(x)).numpy(), expected, rtol=1e-10)


def test_rms_norm_layer_matches_reference():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((3, 5))
    layer = nn.RMSNorm(5, dtype="float64")
    np.testing.assert_allclose(
        layer(_t(x)).numpy(), _rms_norm_np(x, layer.eps), rtol=1e-10
    )


def test_norm_layer_parameter_counts():
    # LayerNorm learns a scale and a shift; RMSNorm only a gain.
    assert len(nn.LayerNorm(5, dtype="float64").parameters()) == 2
    assert len(nn.RMSNorm(5, dtype="float64").parameters()) == 1


@pytest.mark.parametrize("factory", [nn.LayerNorm, nn.RMSNorm])
def test_norm_layers_drop_parameters_when_affine_disabled(factory):
    layer = factory(5, elementwise_affine=False, dtype="float64")
    assert len(layer.parameters()) == 0
    assert layer.num_parameters() == 0


@pytest.mark.parametrize("factory", [nn.LayerNorm, nn.RMSNorm])
def test_norm_layers_reject_mismatched_trailing_dimension(factory):
    layer = factory(5, dtype="float64")
    with pytest.raises(Exception):
        layer(_t(np.zeros((3, 6))))


@pytest.mark.parametrize("factory", [nn.LayerNorm, nn.RMSNorm])
def test_norm_layers_state_dict_round_trip(factory):
    layer = factory(4, dtype="float64")
    before = layer(_t(np.arange(8).reshape(2, 4))).numpy()
    layer.load_state_dict(layer.state_dict())
    after = layer(_t(np.arange(8).reshape(2, 4))).numpy()
    np.testing.assert_allclose(after, before)


# --------------------------------------------------------------------------
# nn.MultiheadAttention
# --------------------------------------------------------------------------


def _mha_np(x_q, x_k, x_v, params, num_heads, causal=False, mask=None, bias=True):
    """Reference multi-head attention following the layer's parameter order."""
    w_q, w_k, w_v, w_o = (p.numpy() for p in params[:4])
    if bias:
        b_q, b_k, b_v, b_o = (p.numpy() for p in params[4:8])
    else:
        b_q = b_k = b_v = b_o = np.zeros(w_q.shape[0])

    batch, len_q, embed = x_q.shape
    len_k = x_k.shape[1]
    head_dim = embed // num_heads

    def split(t, length):
        return t.reshape(batch, length, num_heads, head_dim).transpose(0, 2, 1, 3)

    q = split(x_q @ w_q.T + b_q, len_q)
    k = split(x_k @ w_k.T + b_k, len_k)
    v = split(x_v @ w_v.T + b_v, len_k)

    s = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
    if mask is not None:
        s = s + mask
    if causal:
        i = np.arange(len_q)[:, None]
        j = np.arange(len_k)[None, :]
        s = np.where((j - i) > (len_k - len_q), -np.inf, s)
    s = s - s.max(axis=-1, keepdims=True)
    a = np.exp(s)
    a = a / a.sum(axis=-1, keepdims=True)

    merged = (a @ v).transpose(0, 2, 1, 3).reshape(batch, len_q, embed)
    return merged @ w_o.T + b_o


def test_mha_self_attention_matches_reference():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 6, 8))
    mha = nn.MultiheadAttention(8, 2, dtype="float64")

    out = mha(_t(x))

    assert tuple(out.shape) == (2, 6, 8)
    np.testing.assert_allclose(
        out.numpy(), _mha_np(x, x, x, mha.parameters(), 2), rtol=1e-10
    )


def test_mha_actually_splits_heads():
    # Guards the reshape/transpose: a wrong head layout still yields the right
    # output shape, so compare against a deliberately wrong head count too.
    rng = np.random.default_rng(4)
    x = rng.standard_normal((2, 6, 8))
    mha = nn.MultiheadAttention(8, 2, dtype="float64")
    out = mha(_t(x)).numpy()

    single_head = _mha_np(x, x, x, mha.parameters(), 1)
    assert np.abs(out - single_head).max() > 1e-6


def test_mha_exposes_head_geometry():
    mha = nn.MultiheadAttention(12, 3, dtype="float64")
    assert (mha.embed_dim, mha.num_heads, mha.head_dim) == (12, 3, 4)
    assert mha.is_causal is False
    assert len(mha.parameters()) == 8


def test_mha_causal_matches_reference():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((2, 6, 8))
    mha = nn.MultiheadAttention(8, 2, is_causal=True, dtype="float64")
    assert mha.is_causal is True
    np.testing.assert_allclose(
        mha(_t(x)).numpy(),
        _mha_np(x, x, x, mha.parameters(), 2, causal=True),
        rtol=1e-10,
    )


def test_mha_causal_hides_future_tokens_from_earlier_positions():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((1, 5, 8))
    mha = nn.MultiheadAttention(8, 2, is_causal=True, dtype="float64")
    baseline = mha(_t(x)).numpy()

    perturbed_input = x.copy()
    perturbed_input[:, -1, :] += 5.0
    perturbed = mha(_t(perturbed_input)).numpy()

    np.testing.assert_allclose(perturbed[:, 0, :], baseline[:, 0, :], rtol=1e-12)
    assert np.abs(perturbed[:, -1, :] - baseline[:, -1, :]).max() > 1e-9


def test_mha_cross_attention_follows_query_length():
    rng = np.random.default_rng(7)
    q = rng.standard_normal((2, 3, 8))
    kv = rng.standard_normal((2, 7, 8))
    mha = nn.MultiheadAttention(8, 2, dtype="float64")

    out = mha.forward_qkv(_t(q), _t(kv), _t(kv))

    assert tuple(out.shape) == (2, 3, 8)
    np.testing.assert_allclose(
        out.numpy(), _mha_np(q, kv, kv, mha.parameters(), 2), rtol=1e-10
    )


def test_mha_cross_attention_applies_float_mask():
    rng = np.random.default_rng(8)
    q = rng.standard_normal((2, 3, 8))
    kv = rng.standard_normal((2, 5, 8))
    mask = rng.standard_normal((3, 5))
    mha = nn.MultiheadAttention(8, 2, dtype="float64")

    out = mha.forward_qkv(_t(q), _t(kv), _t(kv), attn_mask=_t(mask))

    np.testing.assert_allclose(
        out.numpy(), _mha_np(q, kv, kv, mha.parameters(), 2, mask=mask), rtol=1e-10
    )


def test_mha_without_bias_has_only_projection_weights():
    rng = np.random.default_rng(9)
    x = rng.standard_normal((2, 4, 8))
    mha = nn.MultiheadAttention(8, 2, bias=False, dtype="float64")

    assert len(mha.parameters()) == 4
    np.testing.assert_allclose(
        mha(_t(x)).numpy(),
        _mha_np(x, x, x, mha.parameters(), 2, bias=False),
        rtol=1e-10,
    )


def test_mha_rejects_embed_dim_not_divisible_by_heads():
    with pytest.raises(Exception):
        nn.MultiheadAttention(10, 4, dtype="float64")


def test_mha_rejects_non_batch_first_input():
    mha = nn.MultiheadAttention(8, 2, dtype="float64")
    with pytest.raises(Exception):
        mha(_t(np.zeros((4, 8))))


def test_mha_rejects_key_value_length_mismatch():
    mha = nn.MultiheadAttention(8, 2, dtype="float64")
    q = _t(np.zeros((2, 3, 8)))
    with pytest.raises(Exception):
        mha.forward_qkv(q, _t(np.zeros((2, 5, 8))), _t(np.zeros((2, 4, 8))))


def test_mha_gradients_reach_every_parameter_and_the_input():
    rng = np.random.default_rng(10)
    x = _t(rng.standard_normal((1, 3, 4)), requires_grad=True)
    mha = nn.MultiheadAttention(4, 2, dtype="float64")

    mha(x).sum().backward()

    assert all(p.grad is not None for p in mha.parameters())
    assert x.grad is not None


# --------------------------------------------------------------------------
# Integration
# --------------------------------------------------------------------------


def test_pre_norm_transformer_block_trains_end_to_end():
    """Embedding -> RMSNorm -> causal attention -> residual, as in a real LLM."""
    dim = 8
    emb = nn.Embedding(20, dim, dtype="float64")
    norm = nn.RMSNorm(dim, dtype="float64")
    attn = nn.MultiheadAttention(dim, 2, is_causal=True, dtype="float64")

    hidden = emb(_ids([[1, 5, 9, 2]]))
    out = hidden + attn(norm(hidden))
    assert tuple(out.shape) == (1, 4, dim)

    out.sum().backward()
    assert emb.parameters()[0].grad is not None
    assert all(p.grad is not None for p in attn.parameters())
    assert all(p.grad is not None for p in norm.parameters())


def test_norm_layer_composes_inside_sequential():
    rng = np.random.default_rng(12)
    model = nn.Sequential([nn.LayerNorm(4, dtype="float64"), nn.ReLU()])
    out = model(_t(rng.standard_normal((2, 4)))).numpy()
    assert out.shape == (2, 4)
    assert (out >= 0).all()
