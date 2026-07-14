# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

from minitensor import nn
from minitensor.nn import (
    BatchNorm1d,
    BatchNorm2d,
    DenseLayer,
    Dropout,
    Dropout2d,
    Sequential,
)
from minitensor.tensor import Tensor


def test_relu_forward():
    x = Tensor.from_numpy(np.array([-1.0, 0.0, 1.0], dtype=np.float32))
    layer = nn.ReLU()
    y = layer.forward(x._tensor)
    np.testing.assert_allclose(y.numpy(), [0.0, 0.0, 1.0])


def test_elu_forward():
    x = Tensor.from_numpy(np.array([-1.0, 0.0, 1.0], dtype=np.float32))
    layer = nn.ELU(alpha=0.5)
    y = layer.forward(x._tensor)
    expected = [0.5 * (np.exp(-1.0) - 1.0), 0.0, 1.0]
    np.testing.assert_allclose(y.numpy(), expected, rtol=1e-5)


def test_gelu_forward():
    x = Tensor.from_numpy(np.array([-1.0, 0.0, 1.0], dtype=np.float32))
    layer = nn.GELU()
    y = layer.forward(x._tensor)
    arr = x.numpy()
    expected = (
        0.5 * arr * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (arr + 0.044715 * arr**3)))
    )
    np.testing.assert_allclose(y.numpy(), expected, rtol=1e-5, atol=1e-5)


def test_batchnorm1d_train_eval_behavior():
    layer = BatchNorm1d(2)
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    out = layer.forward(x._tensor)
    out_np = out.numpy()
    assert np.allclose(out_np.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(out_np.var(axis=0), 1.0, atol=1e-5)

    layer.eval()
    x2 = Tensor([[5.0, 6.0], [7.0, 8.0]], dtype="float32")
    out2 = layer.forward(x2._tensor)
    assert out2.shape == list(x2.shape)


def test_batchnorm2d_channel_normalization():
    layer = BatchNorm2d(3)
    x = Tensor(np.random.randn(2, 3, 4, 4).astype("float32"))
    out = layer.forward(x._tensor)
    out_np = out.numpy()
    channel_var = out_np.var(axis=(0, 2, 3))
    assert np.all(channel_var > 0)


def test_dense_layer_forward_and_stats():
    layer = DenseLayer(3, 2)
    x = Tensor.rand([4, 3])
    y = layer.forward(x._tensor)
    assert y.shape == [4, 2]
    stats = layer.parameter_stats()
    assert stats["total_parameters"] == 3 * 2 + 2
    mem = layer.memory_usage()
    assert mem["total_bytes"] == (3 * 2 + 2) * 4
    assert mem["bytes_by_dtype"]["Float32"] == (3 * 2 + 2) * 4
    summary = layer.summary()
    assert "Total Parameters" in summary


def test_dense_layer_forward_shape_error():
    layer = DenseLayer(3, 2)
    bad_input = Tensor.rand([4, 4])
    with pytest.raises(ValueError):
        layer.forward(bad_input._tensor)


def test_dropout_noop_when_p_zero():
    layer = Dropout(0.0)
    x = Tensor.ones([2, 3])
    y = layer.forward(x._tensor)
    assert np.allclose(y.numpy(), x.numpy())


def test_dropout2d_all_zero_when_p_one():
    layer = Dropout2d(1.0)
    x = Tensor.ones([1, 2, 4, 4])
    y = layer.forward(x._tensor)
    assert np.allclose(y.numpy(), 0.0)


ERROR_NESTED_SEQUENTIAL = "Nested Sequential modules are not supported"


def test_sequential_rejects_nested_sequential_modules_cleanly():
    inner = Sequential([DenseLayer(3, 4)])

    with pytest.raises(TypeError, match=ERROR_NESTED_SEQUENTIAL):
        Sequential([DenseLayer(3, 4), inner])


def test_sequential_add_module_rejects_nested_sequential_modules_cleanly():
    outer = Sequential()
    inner = Sequential([DenseLayer(3, 4)])

    with pytest.raises(TypeError, match=ERROR_NESTED_SEQUENTIAL):
        outer.add_module("nested", inner)


def test_sequential_add_module_failure_does_not_mutate_existing_modules():
    outer = Sequential([DenseLayer(2, 3)])
    base_params = outer.num_parameters()

    with pytest.raises(TypeError, match=ERROR_NESTED_SEQUENTIAL):
        outer.add_module("nested", Sequential([DenseLayer(3, 4)]))

    assert outer.num_parameters() == base_params


def test_sequential_add_module_still_accepts_valid_layer_after_failed_insert():
    outer = Sequential([DenseLayer(2, 3)])

    with pytest.raises(TypeError, match=ERROR_NESTED_SEQUENTIAL):
        outer.add_module("nested", Sequential([DenseLayer(3, 4)]))

    outer.add_module("next", DenseLayer(3, 5))
    assert outer.num_parameters() == (2 * 3 + 3) + (3 * 5 + 5)


def _conv_finite_diff(build, arr, weights, eps=1e-2):
    grad = np.zeros_like(arr)
    it = np.nditer(arr, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        plus = arr.copy()
        plus[i] += eps
        minus = arr.copy()
        minus[i] -= eps
        grad[i] = (
            (build(plus).numpy() * weights).sum()
            - (build(minus).numpy() * weights).sum()
        ) / (2 * eps)
    return grad


@pytest.mark.parametrize(
    "stride,padding,xshape,wshape",
    [
        ((1, 1), (0, 0), (2, 2, 5, 5), (3, 2, 3, 3)),
        ((1, 1), (1, 1), (1, 2, 5, 5), (2, 2, 3, 3)),
        ((2, 2), (1, 1), (1, 1, 6, 6), (2, 1, 3, 3)),
    ],
)
def test_conv2d_is_differentiable(stride, padding, xshape, wshape):
    # Regression: Conv2d forward was correct but wired no backward, so a CNN
    # could not be trained. Check input/weight/bias gradients against finite
    # differences.
    import minitensor as mt

    rng = np.random.default_rng(0)
    x = rng.standard_normal(xshape).astype(np.float32)
    w = rng.standard_normal(wshape).astype(np.float32)
    b = rng.standard_normal(wshape[0]).astype(np.float32)

    def conv(xn=None, wn=None, bn=None):
        return mt.nn.conv2d(
            Tensor((x if xn is None else xn).tolist(), dtype="float32"),
            Tensor((w if wn is None else wn).tolist(), dtype="float32"),
            Tensor((b if bn is None else bn).tolist(), dtype="float32"),
            stride,
            padding,
        )

    mx = Tensor(x.tolist(), dtype="float32", requires_grad=True)
    mw = Tensor(w.tolist(), dtype="float32", requires_grad=True)
    mb = Tensor(b.tolist(), dtype="float32", requires_grad=True)
    out = mt.nn.conv2d(mx, mw, mb, stride, padding)
    weights = np.random.default_rng(1).standard_normal(out.numpy().shape)
    (out * Tensor(weights.tolist(), dtype="float32")).sum().backward()

    gx = _conv_finite_diff(lambda a: conv(xn=a), x, weights)
    gw = _conv_finite_diff(lambda a: conv(wn=a), w, weights)
    gb = _conv_finite_diff(lambda a: conv(bn=a), b, weights)
    for name, ana, num in [
        ("x", mx.grad.numpy(), gx),
        ("w", mw.grad.numpy(), gw),
        ("b", mb.grad.numpy(), gb),
    ]:
        rel = np.abs(ana - num).max() / (np.abs(num).max() + 1e-6)
        assert rel < 1e-2, f"conv {name} grad relerr {rel}"


def test_cnn_training_reduces_loss():
    import minitensor as mt

    rng = np.random.default_rng(0)
    n = 48
    x = rng.standard_normal((n, 1, 8, 8)).astype(np.float32)
    y = (x[:, 0, :4, :4].sum((1, 2)) > 0).astype(np.int64)
    conv = mt.nn.Conv2d(1, 4, 3, padding=1)
    fc = mt.nn.DenseLayer(4 * 8 * 8, 2)
    params = list(conv.parameters()) + list(fc.parameters())
    opt = mt.optim.Adam(params, lr=0.01)
    mx = Tensor(x.tolist(), dtype="float32")
    my = Tensor(y.tolist(), dtype="int64")
    first = last = None
    for _ in range(40):
        opt.zero_grad()
        h = conv(mx).relu().reshape(n, -1)
        loss = mt.nn.cross_entropy(fc(h), my, "mean")
        loss.backward()
        opt.step()
        val = float(loss.numpy())
        first = val if first is None else first
        last = val
    assert last < first * 0.6


def test_reduced_loss_is_scalar():
    import minitensor as mt

    pred = Tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float64")
    tgt = Tensor([[1.5, 2.5], [2.5, 3.5]], dtype="float64")
    loss = mt.nn.mse_loss(pred, tgt, "mean")
    assert loss.numpy().shape == ()
    assert float(loss.numpy()) == pytest.approx(0.25)
