# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import numpy as np
import pytest

import minitensor as mt
from minitensor import functional as F
from minitensor import nn
from minitensor.tensor import Tensor


def test_relu_negative_and_nan():
    t = mt.Tensor([-1.0, float("nan"), 2.0])
    out = t.relu()
    vals = out.numpy()
    np.testing.assert_allclose(vals, np.array([0.0, np.nan, 2.0]), equal_nan=True)


def test_batch_norm_running_var_is_unbiased():
    # PyTorch stores the UNBIASED batch variance in running_var (Bessel's
    # correction, var_biased * n/(n-1)) even though the normalization itself
    # uses the biased estimate. For x with 2 rows per channel the correction
    # factor is 2/(2-1) = 2. The running buffers are updated in place, so they
    # are passed as raw engine tensors.
    x = mt.Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32"))
    running_mean = mt.Tensor(np.array([0.0, 0.0], dtype="float32"))
    running_var = mt.Tensor(np.array([1.0, 1.0], dtype="float32"))

    F.batch_norm(x, running_mean, running_var, training=True, momentum=0.1)

    # Per channel over N=2: mean=[2,3], biased var=[1,1], unbiased=[2,2].
    #   running_mean = 0.9*0 + 0.1*[2,3] = [0.2, 0.3]
    #   running_var  = 0.9*1 + 0.1*[2,2] = [1.1, 1.1]  (biased form -> 1.0)
    np.testing.assert_allclose(running_mean.numpy(), [0.2, 0.3], rtol=1e-5)
    np.testing.assert_allclose(running_var.numpy(), [1.1, 1.1], rtol=1e-5)


def test_hardshrink_matches_numpy_and_grad():
    data = np.array([-1.2, -0.25, 0.0, 0.35, 0.8], dtype=np.float32)
    lambd = 0.3
    tensor = mt.Tensor(data, requires_grad=True)

    out = tensor.hardshrink(lambd=lambd)
    expected = np.where((data > lambd) | (data < -lambd), data, 0.0)
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6)

    out.sum().backward()
    expected_grad = np.where((data > lambd) | (data < -lambd), 1.0, 0.0)
    np.testing.assert_allclose(tensor.grad.numpy(), expected_grad, rtol=1e-6)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_hardshrink_propagates_nan(dtype):
    # hardshrink must pass NaN through, like every other elementwise activation
    # (and PyTorch). A NaN input compares false against both bounds; testing
    # the finite-value complement `|x| > lambd` would drop NaN into the "zero"
    # branch, silently turning NaN into 0.
    tensor = mt.Tensor([float("nan"), 1.2, -0.2, 0.9], dtype=dtype)
    out = tensor.hardshrink(lambd=0.3).numpy()
    assert np.isnan(out[0])
    np.testing.assert_allclose(
        out[1:], np.array([1.2, 0.0, 0.9], dtype=out.dtype), rtol=1e-6
    )


def test_hardshrink_invalid_lambda():
    t = mt.Tensor([1.0])
    with pytest.raises(ValueError):
        _ = t.hardshrink(lambd=-1.0)


def test_sigmoid_tanh_extreme_inputs():
    t = mt.Tensor([1000.0, -1000.0])
    sig = t.sigmoid()
    tanh = t.tanh()
    np.testing.assert_allclose(sig.numpy(), np.array([1.0, 0.0]), atol=1e-6)
    np.testing.assert_allclose(tanh.numpy(), np.array([1.0, -1.0]), atol=1e-6)


def test_softsign_matches_numpy_and_grad():
    data = np.array([-5.0, -0.5, 0.0, 0.25, 4.0], dtype=np.float32)
    tensor = mt.Tensor(data, requires_grad=True)

    out = tensor.softsign()
    expected = data / (1.0 + np.abs(data))
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6)

    out.sum().backward()
    expected_grad = 1.0 / (1.0 + np.abs(data)) ** 2
    np.testing.assert_allclose(tensor.grad.numpy(), expected_grad, rtol=1e-6)


def test_log_softmax_stability_large_range():
    x = np.array([[1000.0, -1000.0, 0.0]], dtype=np.float32)
    t = mt.Tensor(x)
    log_sm = t.log_softmax(dim=1)
    shifted = x - x.max(axis=1, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=1, keepdims=True)) + x.max(
        axis=1, keepdims=True
    )
    expected = x - logsumexp
    np.testing.assert_allclose(log_sm.numpy(), expected, atol=1e-6)
    np.testing.assert_allclose(
        np.exp(log_sm.numpy()).sum(axis=1), np.array([1.0]), atol=1e-6
    )


def test_log_softmax_backward_matches_manual():
    data = np.array([[1.0, -2.0, 0.5], [-0.5, 1.5, -1.0]], dtype=np.float32)
    grad = np.array([[0.2, -0.1, 0.3], [-0.4, 0.25, -0.15]], dtype=np.float32)

    tensor = mt.Tensor(data, requires_grad=True)
    out = tensor.log_softmax(dim=1)
    out.backward(mt.Tensor(grad))

    expected = grad - np.exp(out.numpy()) * grad.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(tensor.grad.numpy(), expected, rtol=1e-6)


def test_log_softmax_scalar_returns_zero():
    t = mt.Tensor(2.5)
    out = t.log_softmax()
    assert out.shape == ()
    assert out.item() == pytest.approx(0.0)


def test_log1p_expm1_matches_numpy():
    values = np.array([-0.5, 0.0, 1.5], dtype=np.float32)
    t = mt.Tensor(values)

    np.testing.assert_allclose(t.log1p().numpy(), np.log1p(values), rtol=1e-6)
    np.testing.assert_allclose(t.expm1().numpy(), np.expm1(values), rtol=1e-6)

    edge = mt.Tensor([-1.0, -2.0], dtype="float32")
    res = edge.log1p().numpy()
    assert np.isneginf(res[0])
    assert np.isnan(res[1])


def test_rsqrt_matches_numpy_and_grad():
    data = np.array([0.25, 1.0, 4.0], dtype=np.float32)
    tensor = mt.Tensor(data, requires_grad=True)

    out = tensor.rsqrt()
    expected = 1.0 / np.sqrt(data)
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6)

    out.sum().backward()
    expected_grad = -0.5 * np.power(data, -1.5)
    np.testing.assert_allclose(tensor.grad.numpy(), expected_grad, rtol=1e-5)


def test_logsumexp_matches_numpy_and_grad():
    data = np.array([[1.0, 1.0, -1.0], [2.0, -2.0, 3.0]], dtype=np.float32)
    tensor = mt.Tensor(data, requires_grad=True)

    out = tensor.logsumexp(dim=1)
    expected = np.log(np.exp(data).sum(axis=1))
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6)

    out.sum().backward()
    expected_grad = np.exp(data - expected[:, None])
    np.testing.assert_allclose(tensor.grad.numpy(), expected_grad, rtol=1e-6)


def test_logsumexp_keepdim_and_all_dims():
    data = np.array([[[1.0, -2.0], [3.0, 0.5]]], dtype=np.float32)
    tensor = mt.Tensor(data)

    keepdim = tensor.logsumexp(dim=(1, 2), keepdim=True)
    expected_keepdim = np.log(np.exp(data).sum(axis=(1, 2), keepdims=True))
    np.testing.assert_allclose(keepdim.numpy(), expected_keepdim, rtol=1e-6)

    collapsed = tensor.logsumexp(dim=(1, 2), keepdim=False)
    expected_collapsed = expected_keepdim.reshape(collapsed.shape)
    np.testing.assert_allclose(collapsed.numpy(), expected_collapsed, rtol=1e-6)

    with pytest.raises(RuntimeError):
        _ = tensor.astype("int32").logsumexp(dim=1)


def test_logsumexp_non_finite_rows():
    # The stable formula max + log(sum(exp(x - max))) degenerates to
    # inf - inf = NaN when the row max is not finite; the correct limits are
    # +inf for rows containing +inf, -inf for all--inf rows, NaN for NaN rows.
    data = np.array(
        [[0.0, -np.inf, 1.0], [-np.inf, -np.inf, -np.inf], [np.inf, 0.0, 1.0]],
        dtype=np.float32,
    )
    out = mt.Tensor(data).logsumexp(dim=1).numpy()
    np.testing.assert_allclose(out[0], np.logaddexp(0.0, 1.0), rtol=1e-6)
    assert out[1] == -np.inf
    assert out[2] == np.inf

    nan_row = mt.Tensor(np.array([[np.nan, 1.0]], dtype=np.float32))
    assert np.isnan(nan_row.logsumexp(dim=1).numpy()).all()

    data64 = np.array([[-np.inf, -np.inf], [np.inf, 1.0]], dtype=np.float64)
    out64 = mt.Tensor(data64).logsumexp(dim=1).numpy()
    assert out64[0] == -np.inf and out64[1] == np.inf

    multi = np.full((2, 2, 2), -np.inf, dtype=np.float32)
    multi[0] = 1.0
    out_multi = mt.Tensor(multi).logsumexp(dim=(1, 2), keepdim=True).numpy().ravel()
    np.testing.assert_allclose(out_multi[0], 1.0 + np.log(4.0), rtol=1e-6)
    assert out_multi[1] == -np.inf


def test_activation_methods_have_pytorch_defaults():
    # These pymethods take Option-typed parameters; without an explicit
    # #[pyo3(signature)] PyO3 makes them required, breaking no-arg calls.
    data = np.array([[1.0, -2.0], [3.0, -0.25]], dtype=np.float32)
    t = mt.Tensor(data)

    np.testing.assert_allclose(t.softplus().numpy(), np.log1p(np.exp(data)), rtol=1e-5)
    np.testing.assert_allclose(
        t.elu().numpy(), np.where(data > 0, data, np.expm1(data)), rtol=1e-5
    )
    from math import erf

    np.testing.assert_allclose(
        t.gelu().numpy(),
        0.5 * data * (1.0 + np.vectorize(erf)(data / np.sqrt(2.0))),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        t.hardshrink().numpy(), np.where(np.abs(data) > 0.5, data, 0.0), rtol=1e-6
    )

    stacked = mt.Tensor.stack([t, t])
    assert tuple(stacked.shape) == (2, 2, 2)
    joined = mt.Tensor.concatenate([t, t])
    assert tuple(joined.shape) == (4, 2)
    joined_axis = mt.Tensor.concatenate([t, t], axis=1)
    assert tuple(joined_axis.shape) == (2, 4)


def test_softplus_matches_numpy_and_grad():
    data = np.array([-5.0, 0.0, 5.0, 50.0], dtype=np.float32)
    tensor = mt.Tensor(data, requires_grad=True)

    out = tensor.softplus(beta=1.5, threshold=10.0)
    expected = np.where(
        1.5 * data > 10.0,
        data,
        np.log1p(np.exp(1.5 * data)) / 1.5,
    )
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)

    out.sum().backward()
    expected_grad = np.where(
        1.5 * data > 10.0,
        1.0,
        1.0 / (1.0 + np.exp(-1.5 * data)),
    )
    np.testing.assert_allclose(tensor.grad.numpy(), expected_grad, rtol=1e-5)


def test_logaddexp_matches_numpy_and_grad():
    a_vals = np.array([[1.0, 2.0, -100.0], [0.5, -0.75, 3.0]], dtype=np.float32)
    b_vals = np.array([[0.0, -2.0, -100.0], [1.5, 0.25, -4.0]], dtype=np.float32)

    a = mt.Tensor(a_vals, requires_grad=True)
    b = mt.Tensor(b_vals, requires_grad=True)
    result = a.logaddexp(b)
    expected = np.logaddexp(a_vals, b_vals)
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)

    result.sum().backward()
    expected_grad_a = np.exp(a_vals - expected)
    expected_grad_b = np.exp(b_vals - expected)
    np.testing.assert_allclose(a.grad.numpy(), expected_grad_a, rtol=1e-6)
    np.testing.assert_allclose(b.grad.numpy(), expected_grad_b, rtol=1e-6)


def test_gelu_exact_matches_reference_and_grad():
    data = np.array([-2.5, -0.5, 0.0, 0.75, 3.25], dtype=np.float32)
    tensor = mt.Tensor(data, requires_grad=True)

    out = tensor.gelu(approximate="none")
    sqrt_two = np.sqrt(np.array(2.0, dtype=data.dtype))
    erf_vals = np.array(
        [math.erf(float(v)) for v in (data / sqrt_two)], dtype=data.dtype
    )
    expected = 0.5 * data * (1.0 + erf_vals)
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-7)

    out.sum().backward()
    sqrt_two_pi = np.sqrt(np.array(2.0 * np.pi, dtype=data.dtype))
    expected_grad = 0.5 * (1.0 + erf_vals)
    expected_grad += data * np.exp(-0.5 * data**2) / sqrt_two_pi
    np.testing.assert_allclose(tensor.grad.numpy(), expected_grad, rtol=1e-6, atol=1e-7)


def test_gelu_tanh_approximation_grad():
    data = np.array([-1.5, -0.25, 0.5, 1.25], dtype=np.float32)
    tensor = mt.Tensor(data, requires_grad=True)

    out = tensor.gelu(approximate="tanh")
    k = np.sqrt(np.array(2.0 / np.pi, dtype=data.dtype))
    inner = k * (data + 0.044715 * data**3)
    expected = 0.5 * data * (1.0 + np.tanh(inner))
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6)

    out.sum().backward()
    tanh_inner = np.tanh(inner)
    sech2 = 1.0 - tanh_inner**2
    expected_grad = 0.5 * (1.0 + tanh_inner)
    expected_grad += 0.5 * data * sech2 * k * (1.0 + 3.0 * 0.044715 * data**2)
    np.testing.assert_allclose(tensor.grad.numpy(), expected_grad, rtol=1e-6)


def test_elu_forward_and_grad():
    data = np.array([-2.0, -0.1, 0.0, 1.5], dtype=np.float32)
    alpha = 1.3
    tensor = mt.Tensor(data, requires_grad=True)

    out = tensor.elu(alpha=alpha)
    expected = np.where(data > 0.0, data, alpha * (np.exp(data) - 1.0))
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6)

    out.sum().backward()
    expected_grad = np.where(data > 0.0, 1.0, expected + alpha)
    np.testing.assert_allclose(tensor.grad.numpy(), expected_grad, rtol=1e-6)


def test_selu_forward_and_grad():
    data = np.array([-1.0, -0.2, 0.3, 2.0], dtype=np.float32)
    tensor = mt.Tensor(data, requires_grad=True)

    out = tensor.selu()
    scale = np.float32(1.0507009873554805)
    alpha = np.float32(1.6732632423543772)
    expected = np.where(
        data > 0.0,
        scale * data,
        scale * alpha * (np.exp(data) - 1.0),
    )
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6)

    out.sum().backward()
    expected_grad = np.where(data > 0.0, scale, expected + scale * alpha)
    np.testing.assert_allclose(tensor.grad.numpy(), expected_grad, rtol=1e-6)


def test_silu_forward_and_grad():
    data = np.array([-3.0, -0.5, 0.5, 2.5], dtype=np.float32)
    tensor = mt.Tensor(data, requires_grad=True)

    out = tensor.silu()
    sigmoid = 1.0 / (1.0 + np.exp(-data))
    expected = data * sigmoid
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6)

    out.sum().backward()
    expected_grad = sigmoid * (1.0 + data * (1.0 - sigmoid))
    np.testing.assert_allclose(tensor.grad.numpy(), expected_grad, rtol=1e-6)


def test_silu_extreme_values_and_grad_stability():
    values = np.array([-1000.0, 0.0, 1000.0], dtype=np.float32)
    tensor = mt.Tensor(values, requires_grad=True)

    out = tensor.silu()
    expected = values / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))
    np.testing.assert_allclose(out.numpy(), expected, atol=1e-6, rtol=1e-6)

    out.sum().backward()
    grads = tensor.grad.numpy()
    assert np.isfinite(grads).all()
    np.testing.assert_allclose(
        grads, np.array([0.0, 0.5, 1.0], dtype=np.float32), atol=1e-4
    )


def test_functional_softmax_matches_tensor():
    x_np = np.array([[1.0, 2.0], [3.0, 4.0]])
    x = mt.Tensor(x_np.tolist())
    result = F.softmax(x)
    expected = np.exp(x_np - x_np.max(axis=1, keepdims=True))
    expected = expected / expected.sum(axis=1, keepdims=True)
    assert np.allclose(result.numpy(), expected)


def test_functional_softmax_dim():
    x_np = np.array([[1.0, 2.0], [3.0, 4.0]])
    x = mt.Tensor(x_np.tolist())
    result = F.softmax(x, dim=0)
    expected = np.exp(x_np - x_np.max(axis=0, keepdims=True))
    expected = expected / expected.sum(axis=0, keepdims=True)
    assert np.allclose(result.numpy(), expected)


def test_softmax_extreme_values():
    x_np = np.array([[1e9, -1e9], [-1e9, 1e9]], dtype=np.float32)
    x = mt.Tensor(x_np.tolist())
    result = F.softmax(x)
    shifted = x_np - x_np.max(axis=1, keepdims=True)
    expected = np.exp(shifted)
    expected = expected / expected.sum(axis=1, keepdims=True)
    assert np.allclose(result.numpy(), expected)
    assert np.allclose(result.numpy().sum(axis=1), np.array([1.0, 1.0]))


def test_softmax_scalar_returns_one():
    x = mt.Tensor(3.5)
    result = F.softmax(x)
    assert result.shape == ()
    assert result.item() == pytest.approx(1.0)


def test_softmax_empty_dim_returns_empty():
    x = mt.Tensor.zeros((2, 0, 3), dtype="float32")
    result = F.softmax(x, dim=1)
    assert result.shape == (2, 0, 3)
    assert result.numel() == 0


def test_masked_softmax_matches_expected():
    x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mask_np = np.array([[True, False], [False, True]])
    x = mt.Tensor(x_np.tolist())
    mask = mt.Tensor(mask_np.tolist(), dtype="bool")
    result = F.masked_softmax(x, mask, dim=1)
    expected = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    assert np.allclose(result.numpy(), expected)


def test_masked_softmax_broadcasts_mask():
    x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mask_np = np.array([[True], [False]])
    x = mt.Tensor(x_np.tolist())
    mask = mt.Tensor(mask_np.tolist(), dtype="bool")
    result = F.masked_softmax(x, mask, dim=1)
    expected = np.array([[0.0, 0.0], [0.26894143, 0.7310586]], dtype=np.float32)
    assert np.allclose(result.numpy(), expected, atol=1e-6)


def test_masked_log_softmax_all_masked_is_neg_inf():
    x_np = np.array([1.0, 2.0], dtype=np.float32)
    mask_np = np.array([True, True])
    x = mt.Tensor(x_np.tolist())
    mask = mt.Tensor(mask_np.tolist(), dtype="bool")
    result = F.masked_log_softmax(x, mask, dim=0)
    out = result.numpy()
    assert np.all(np.isneginf(out))


def test_masked_softmax_all_neg_inf_unmasked_is_zero():
    x_np = np.array([-np.inf, -np.inf], dtype=np.float32)
    mask_np = np.array([False, False])
    x = mt.Tensor(x_np.tolist())
    mask = mt.Tensor(mask_np.tolist(), dtype="bool")
    result = F.masked_softmax(x, mask, dim=0)
    assert np.allclose(result.numpy(), np.array([0.0, 0.0], dtype=np.float32))


def _reference_masked_softmax(x, mask, axis, log=False):
    """NumPy reference: masked entries drop out of both the max and the sum."""
    masked_x = np.where(mask, -np.inf, x.astype(np.float64))
    peak = np.max(masked_x, axis=axis, keepdims=True)
    empty = ~np.isfinite(peak)  # the whole slice is masked, or is all -inf
    shift = np.where(empty, 0.0, peak)
    exps = np.where(mask, 0.0, np.exp(masked_x - shift))
    total = exps.sum(axis=axis, keepdims=True)
    if log:
        out = np.where(
            mask, -np.inf, masked_x - (np.log(np.where(total == 0, 1, total)) + shift)
        )
        return np.where(empty, -np.inf, out)
    out = np.where(total > 0, exps / np.where(total == 0, 1, total), 0.0)
    return np.where(empty, 0.0, out)


# The softmax family is one generic kernel per variant rather than one per
# dtype, and the masked variants resolve a possibly-broadcast mask through a
# single closure. Sweep both dtypes, every axis of a rank-1/2/3 tensor, and
# masks that do and do not need broadcasting.
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    "shape,mask_shape",
    [
        ((6,), (6,)),
        ((4, 6), (4, 6)),
        ((4, 6), (1, 6)),
        ((3, 4, 5), (3, 4, 5)),
        ((3, 4, 5), (1, 1, 5)),
        ((3, 4, 5), (1, 4, 5)),
    ],
)
def test_masked_softmax_family_matches_reference(dtype, shape, mask_shape):
    rng = np.random.default_rng(20240717)
    np_dtype = np.float32 if dtype == "float32" else np.float64
    x_np = (rng.standard_normal(shape) * 3).astype(np_dtype)
    mask_np = rng.random(mask_shape) < 0.4
    broadcast_mask = np.broadcast_to(mask_np, shape)

    x = mt.Tensor(x_np)
    mask = mt.Tensor(np.ascontiguousarray(mask_np)).astype("bool")

    for axis in range(len(shape)):
        got = F.masked_softmax(x, mask, dim=axis).numpy()
        want = _reference_masked_softmax(x_np, broadcast_mask, axis)
        assert np.allclose(
            got, want, rtol=1e-5, atol=1e-6
        ), f"masked_softmax axis={axis}"

        got_log = F.masked_log_softmax(x, mask, dim=axis).numpy()
        want_log = _reference_masked_softmax(x_np, broadcast_mask, axis, log=True)
        finite = np.isfinite(want_log)
        assert np.allclose(
            got_log[finite], want_log[finite], rtol=1e-5, atol=1e-5
        ), f"masked_log_softmax axis={axis}"
        assert np.all(np.isneginf(got_log[~finite])), f"masked_log_softmax axis={axis}"


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("shape", [(6,), (4, 7), (3, 4, 5)])
def test_softmax_family_matches_reference(dtype, shape):
    rng = np.random.default_rng(20240718)
    np_dtype = np.float32 if dtype == "float32" else np.float64
    x_np = (rng.standard_normal(shape) * 3).astype(np_dtype)
    x = mt.Tensor(x_np)

    for axis in range(len(shape)):
        shifted = x_np.astype(np.float64) - np.max(x_np, axis=axis, keepdims=True)
        want = np.exp(shifted) / np.exp(shifted).sum(axis=axis, keepdims=True)
        assert np.allclose(F.softmax(x, dim=axis).numpy(), want, rtol=1e-5, atol=1e-6)
        assert np.allclose(
            F.log_softmax(x, dim=axis).numpy(), np.log(want), rtol=1e-5, atol=1e-5
        )


def _layer_norm_reference(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    normalized_shape = tuple(normalized_shape)
    dims = len(normalized_shape)
    axis = tuple(range(x.ndim - dims, x.ndim))
    mean = x.mean(axis=axis, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=axis, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    normalized = (x - mean) * inv_std

    output = normalized
    if weight is not None:
        reshape = (1,) * (x.ndim - dims) + weight.shape
        output = output * weight.reshape(reshape)
    if bias is not None:
        reshape = (1,) * (x.ndim - dims) + bias.shape
        output = output + bias.reshape(reshape)

    return output, normalized, inv_std


def test_layer_norm_forward_matches_numpy():
    x = np.array([[1.2, -0.5, 2.0], [0.7, -1.3, 0.25]], dtype=np.float32)
    weight = np.array([1.5, 0.75, -0.25], dtype=np.float32)
    bias = np.array([0.1, -0.2, 0.05], dtype=np.float32)
    eps = 1e-5

    ref, _, _ = _layer_norm_reference(x, (3,), weight=weight, bias=bias, eps=eps)

    tensor = mt.Tensor(x, requires_grad=True)
    weight_tensor = mt.Tensor(weight, requires_grad=True)
    bias_tensor = mt.Tensor(bias, requires_grad=True)

    out = tensor.layer_norm((3,), weight=weight_tensor, bias=bias_tensor, eps=eps)
    np.testing.assert_allclose(out.numpy(), ref, rtol=1e-6, atol=1e-6)


def test_layer_norm_gradients_match_manual_formula():
    x = np.array([[1.2, -0.5, 2.0], [0.7, -1.3, 0.25]], dtype=np.float32)
    weight = np.array([1.5, 0.75, -0.25], dtype=np.float32)
    bias = np.array([0.1, -0.2, 0.05], dtype=np.float32)
    eps = 1e-5

    tensor = mt.Tensor(x, requires_grad=True)
    weight_tensor = mt.Tensor(weight, requires_grad=True)
    bias_tensor = mt.Tensor(bias, requires_grad=True)

    out = tensor.layer_norm((3,), weight=weight_tensor, bias=bias_tensor, eps=eps)
    out.sum().backward()

    grad_output = np.ones_like(x, dtype=np.float32)
    _, normalized, inv_std = _layer_norm_reference(
        x, (3,), weight=weight, bias=bias, eps=eps
    )

    grad_output_hat = grad_output * weight.reshape(1, -1)
    m = float(np.prod((3,), dtype=np.int64))
    sum_grad = grad_output_hat.sum(axis=-1, keepdims=True)
    sum_grad_norm = (grad_output_hat * normalized).sum(axis=-1, keepdims=True)
    grad_input_expected = (
        (grad_output_hat * m - sum_grad - normalized * sum_grad_norm) * inv_std / m
    )
    grad_weight_expected = (grad_output * normalized).sum(axis=0)
    grad_bias_expected = grad_output.sum(axis=0)

    np.testing.assert_allclose(
        tensor.grad.numpy(), grad_input_expected, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        weight_tensor.grad.numpy(), grad_weight_expected, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        bias_tensor.grad.numpy(), grad_bias_expected, rtol=1e-6, atol=1e-6
    )


def test_layer_norm_requires_normalized_shape():
    t = mt.Tensor([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        _ = t.layer_norm(())


def test_batch_norm_training_updates_stats():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    running_mean = Tensor([0.0, 0.0], dtype="float32")
    running_var = Tensor([0.0, 0.0], dtype="float32")
    out = F.batch_norm(x, running_mean, running_var, training=True)
    out_np = out.numpy()
    assert np.allclose(out_np.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(out_np.var(axis=0, ddof=0), 1.0, atol=1e-5)
    assert not np.allclose(running_mean.numpy(), [0.0, 0.0])
    assert not np.allclose(running_var.numpy(), [0.0, 0.0])


def test_batch_norm_eval_uses_running_stats():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    running_mean = Tensor([2.0, 3.0], dtype="float32")
    running_var = Tensor([1.0, 4.0], dtype="float32")
    out = F.batch_norm(x, running_mean, running_var, training=False)
    expected = (x.numpy() - running_mean.numpy()) / np.sqrt(running_var.numpy() + 1e-5)
    assert np.allclose(out.numpy(), expected, atol=1e-5)


def test_batch_norm_with_weight_and_bias():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    running_mean = Tensor([0.0, 0.0], dtype="float32")
    running_var = Tensor([1.0, 1.0], dtype="float32")
    weight = Tensor([1.5, 0.5], dtype="float32")
    bias = Tensor([0.5, -1.0], dtype="float32")
    out = F.batch_norm(x, running_mean, running_var, weight, bias, training=True)
    x_np = x.numpy()
    mean = x_np.mean(axis=0)
    var = x_np.var(axis=0)
    expected = ((x_np - mean) / np.sqrt(var + 1e-5)) * weight.numpy() + bias.numpy()
    assert np.allclose(out.numpy(), expected, atol=1e-5)


def test_batch_norm_zero_variance():
    x = Tensor([[5.0, 5.0], [5.0, 5.0]], dtype="float32")
    running_mean = Tensor([0.0, 0.0], dtype="float32")
    running_var = Tensor([1.0, 1.0], dtype="float32")
    out = F.batch_norm(x, running_mean, running_var, training=True)
    assert np.allclose(out.numpy(), 0.0, atol=1e-5)


def test_dropout_extreme_probabilities():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    out_same = F.dropout(x, p=0.0, training=True)
    assert np.array_equal(out_same.numpy(), x.numpy())
    out_zero = F.dropout(x, p=1.0, training=True)
    assert np.allclose(out_zero.numpy(), 0.0)
    out_eval = F.dropout(x, p=0.5, training=False)
    assert np.array_equal(out_eval.numpy(), x.numpy())


def test_cross_entropy_matches_numpy():
    x_np = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    target_np = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    x = mt.Tensor(x_np.tolist())
    target = mt.Tensor(target_np.tolist())

    loss = F.cross_entropy(x, target, reduction="mean")

    # Manual cross entropy averaged over batch
    shifted = x_np - x_np.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    softmax = exp / exp.sum(axis=1, keepdims=True)
    expected = -(target_np * np.log(softmax)).sum(axis=1).mean()

    assert np.allclose(loss.numpy(), expected)


def test_cross_entropy_dim_argument():
    x_np = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32)
    target_np = np.array([2, 0], dtype=np.int64)
    x = mt.Tensor(x_np.tolist())
    target = mt.Tensor(target_np.tolist())

    loss = F.cross_entropy(x, target, reduction="mean", dim=0)

    shifted = x_np - x_np.max(axis=0, keepdims=True)
    exp = np.exp(shifted)
    softmax = exp / exp.sum(axis=0, keepdims=True)
    expected = -np.log(softmax[target_np, np.arange(softmax.shape[1])]).mean()

    assert np.allclose(loss.numpy(), expected)


def test_cross_entropy_negative_dim():
    x_np = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32)
    target_np = np.array([2, 0], dtype=np.int64)
    x = mt.Tensor(x_np.tolist())
    target = mt.Tensor(target_np.tolist())

    loss = F.cross_entropy(x, target, reduction="mean", dim=-2)

    shifted = x_np - x_np.max(axis=0, keepdims=True)
    exp = np.exp(shifted)
    softmax = exp / exp.sum(axis=0, keepdims=True)
    expected = -np.log(softmax[target_np, np.arange(softmax.shape[1])]).mean()

    assert np.allclose(loss.numpy(), expected)


def test_cross_entropy_invalid_dim_raises():
    x_np = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32)
    target_np = np.array([2, 0], dtype=np.int64)
    x = mt.Tensor(x_np.tolist())
    target = mt.Tensor(target_np.tolist())

    try:
        F.cross_entropy(x, target, dim=2)
    except IndexError:
        pass
    else:
        assert False, "Expected IndexError for invalid dim"


def test_cross_entropy_no_reduction_shape_and_values():
    x_np = np.array(
        [
            [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]],
            [[2.0, 1.0, 0.0, -1.0], [2.0, 1.0, 0.0, -1.0], [2.0, 1.0, 0.0, -1.0]],
        ],
        dtype=np.float32,
    )
    target_np = np.array([[0, 1, 2, 1], [2, 0, 1, 2]], dtype=np.int64)
    x = mt.Tensor(x_np.tolist())
    target = mt.Tensor(target_np.tolist())

    loss = F.cross_entropy(x, target, reduction="none", dim=1)

    shifted = x_np - x_np.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    softmax = exp / exp.sum(axis=1, keepdims=True)
    gathered = np.take_along_axis(softmax, target_np[:, None, :], axis=1).squeeze(1)
    expected = -np.log(gathered)

    assert loss.shape == expected.shape
    assert np.allclose(loss.numpy(), expected)


def test_cross_entropy_no_reduction_backward_scales_each_sample():
    logits = mt.Tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)
    targets = mt.Tensor([0, 1], dtype="int64")
    losses = F.cross_entropy(logits, targets, reduction="none")

    losses.backward(mt.Tensor([2.0, 3.0]))

    np.testing.assert_allclose(
        logits.grad.numpy(),
        np.array([[-1.0, 1.0], [1.5, -1.5]], dtype=np.float32),
        rtol=1e-6,
    )


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize(
    "shape,dim,tshape",
    [
        ((2, 4, 3), 1, (2, 3)),  # class dim in the middle (needs a transpose)
        ((4, 2, 3), 0, (2, 3)),  # class dim first (two transposes)
        ((2, 3, 4), 2, (2, 3)),  # class dim already last (no transpose)
        ((2, 5, 3, 4), 1, (2, 3, 4)),  # 4-D, class dim in the middle
    ],
)
def test_cross_entropy_backward_multidim_matches_analytic(
    reduction, shape, dim, tshape
):
    # Regression: ``F.cross_entropy`` with ``dim`` other than the last one
    # permutes the logits internally. The reshape used for that permutation has
    # to record an autograd node, or the analytical backward's gradient is
    # routed into the pre-reshape (transpose) node and either crashes with an
    # out-of-bounds transpose or silently corrupts the result. Check the
    # gradient against the closed form ``(softmax - onehot) * scale`` for every
    # reduction and class-dimension placement.
    rng = np.random.default_rng(0)
    z = rng.standard_normal(shape).astype(np.float32)
    ti = rng.integers(0, shape[dim], size=tshape).astype(np.int64)

    logits = mt.Tensor(z.tolist(), requires_grad=True)
    target = mt.Tensor(ti.tolist(), dtype="int64")
    loss = F.cross_entropy(logits, target, reduction=reduction, dim=dim)

    if reduction == "none":
        loss.backward(mt.Tensor(np.ones(loss.shape, dtype=np.float32).tolist()))
    else:
        loss.backward()

    shifted = z - z.max(axis=dim, keepdims=True)
    exp = np.exp(shifted)
    softmax = exp / exp.sum(axis=dim, keepdims=True)
    onehot = np.zeros_like(z)
    np.put_along_axis(onehot, np.expand_dims(ti, dim), 1.0, axis=dim)
    scale = (1.0 / int(np.prod(tshape))) if reduction == "mean" else 1.0
    expected_grad = (softmax - onehot) * scale

    grad = logits.grad.numpy()
    assert grad.shape == z.shape
    np.testing.assert_allclose(grad, expected_grad, rtol=1e-5, atol=1e-5)


def test_cross_entropy_soft_target_backward_uses_total_target_mass():
    logits = mt.Tensor([[0.0, 0.0]], requires_grad=True)
    weighted_target = mt.Tensor([[2.0, 0.0]])

    F.cross_entropy(logits, weighted_target, reduction="sum").backward()

    np.testing.assert_allclose(
        logits.grad.numpy(), np.array([[-1.0, 1.0]], dtype=np.float32), rtol=1e-6
    )


def test_cross_entropy_soft_targets_follow_logits_dtype():
    logits = mt.Tensor([[0.0, 0.0]], dtype="float32", requires_grad=True)
    target = mt.Tensor([[1.0, 0.0]], dtype="float64")

    loss = F.cross_entropy(logits, target, reduction="sum")
    loss.backward()

    assert loss.dtype == "float32"
    assert logits.grad.dtype == "float32"
    np.testing.assert_allclose(
        logits.grad.numpy(), np.array([[-0.5, 0.5]], dtype=np.float32), rtol=1e-6
    )


def test_cross_entropy_extreme_logits_remain_finite():
    x_np = np.array([[1000.0, -1000.0]], dtype=np.float32)
    target_np = np.array([[0.0, 1.0]], dtype=np.float32)
    x = mt.Tensor(x_np.tolist())
    target = mt.Tensor(target_np.tolist())
    loss = F.cross_entropy(x, target, reduction="mean")
    # Stable log-softmax preserves the finite log-probability even though
    # materializing softmax itself would underflow the target probability.
    assert np.isfinite(loss.numpy())
    np.testing.assert_allclose(loss.numpy(), 2000.0, rtol=1e-6)


def test_cross_entropy_non_target_negative_infinity_does_not_poison_loss():
    logits = mt.Tensor([[0.0, -np.inf]])

    correct = F.cross_entropy(logits, mt.Tensor([0], dtype="int64"))
    np.testing.assert_array_equal(correct.numpy(), 0.0)

    impossible = F.cross_entropy(logits, mt.Tensor([1], dtype="int64"))
    assert np.isposinf(impossible.numpy())


def test_cross_entropy_confident_correct_prediction_is_finite():
    # A confidently-correct prediction (target == argmax) with a large logit
    # gap must give ~0 loss and a finite gradient. A blanket "mask near-zero
    # probability classes to -inf" made the non-target class's contribution
    # 0 * -inf = NaN, so this used to return NaN for magnitudes >= ~500.
    targets = mt.Tensor([0], dtype="int64")
    for mag in (100.0, 500.0, 1000.0, 10000.0):
        logits = mt.Tensor([[mag, 0.0, -mag]], requires_grad=True)
        loss = F.cross_entropy(logits, targets, reduction="mean")
        assert np.isfinite(loss.numpy()), f"CE not finite at mag={mag}"
        np.testing.assert_allclose(loss.numpy(), 0.0, atol=1e-5)
        loss.backward()
        assert np.all(np.isfinite(logits.grad.numpy()))
        mt.clear_autograd_graph()


def test_bce_loss_positive():
    preds = mt.Tensor([0.8, 0.2]).reshape(2, 1)
    targets = mt.Tensor([1.0, 0.0]).reshape(2, 1)
    loss = nn.BCELoss()(preds, targets)
    assert float(loss.numpy().ravel()[0]) > 0


def _bce_logits_ref(x, t, w=1.0):
    """Stable reference: (1-t)*x + (1+(w-1)*t) * -log(sigmoid(x))."""
    return (1.0 - t) * x + (1.0 + (w - 1.0) * t) * (
        math.log1p(math.exp(-abs(x))) + max(-x, 0.0)
    )


@pytest.mark.parametrize(
    "logit", [-100.0, -50.0, -30.0, -8.0, -1.0, 0.0, 1.0, 30.0, 100.0]
)
@pytest.mark.parametrize("target", [0.0, 1.0, 0.3])
def test_bce_with_logits_matches_reference(logit, target):
    x = mt.Tensor([[logit]], requires_grad=True)
    loss = F.binary_cross_entropy_with_logits(x, mt.Tensor([[target]]), reduction="sum")
    expected = _bce_logits_ref(logit, target)
    np.testing.assert_allclose(loss.numpy().ravel()[0], expected, rtol=1e-5, atol=1e-6)

    # d/dx = sigmoid(x) - target, exact at every magnitude.
    loss.backward()
    sigmoid = (
        1.0 / (1.0 + math.exp(-logit))
        if logit >= 0
        else (math.exp(logit) / (1.0 + math.exp(logit)))
    )
    np.testing.assert_allclose(
        x.grad.numpy().ravel()[0], sigmoid - target, rtol=1e-5, atol=1e-7
    )
    mt.clear_autograd_graph()


@pytest.mark.parametrize("logit", [-30.0, -50.0, -100.0])
def test_bce_with_logits_keeps_gradient_where_sigmoid_bce_loses_it(logit):
    """The reason this loss exists.

    A large negative logit against a target of 1 is a confident, completely
    wrong prediction, and should produce the strongest gradient the loss can
    give: -1. Computing sigmoid first lets it round to 0 in f32, and the
    gradient collapses -- at -30 the two-step path returns about -0.09, and
    past -50 exactly 0, so the example stops contributing anything.
    """
    fused = mt.Tensor([[logit]], requires_grad=True)
    F.binary_cross_entropy_with_logits(
        fused, mt.Tensor([[1.0]]), reduction="sum"
    ).backward()
    np.testing.assert_allclose(fused.grad.numpy().ravel()[0], -1.0, atol=1e-6)
    mt.clear_autograd_graph()

    split = mt.Tensor([[logit]], requires_grad=True)
    F.binary_cross_entropy(
        F.sigmoid(split), mt.Tensor([[1.0]]), reduction="sum"
    ).backward()
    assert abs(split.grad.numpy().ravel()[0]) < 0.5, (
        "sigmoid+BCE unexpectedly kept its gradient; if that path was fixed, "
        "this test no longer demonstrates anything and should be revisited"
    )
    mt.clear_autograd_graph()


def test_bce_with_logits_agrees_with_sigmoid_bce_when_well_conditioned():
    # Where the two-step path is still numerically valid the two must agree.
    logits = np.linspace(-6.0, 6.0, 25).astype(np.float32).reshape(-1, 1)
    targets = np.tile([[0.0], [1.0]], (13, 1))[:25].astype(np.float32)

    fused = mt.Tensor(logits, requires_grad=True)
    fused_loss = F.binary_cross_entropy_with_logits(fused, mt.Tensor(targets))
    fused_loss.backward()
    fused_grad = fused.grad.numpy().copy()
    mt.clear_autograd_graph()

    split = mt.Tensor(logits, requires_grad=True)
    split_loss = F.binary_cross_entropy(F.sigmoid(split), mt.Tensor(targets))
    split_loss.backward()

    np.testing.assert_allclose(
        fused_loss.numpy(), split_loss.numpy(), rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(fused_grad, split.grad.numpy(), rtol=1e-4, atol=1e-6)
    mt.clear_autograd_graph()


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_bce_with_logits_pos_weight_broadcasts_over_columns(reduction):
    logits = [[-2.0, 0.5, 3.0], [1.0, -1.5, 0.0]]
    targets = [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]
    pos_weight = [0.5, 2.0, 3.0]  # one per column, broadcast across rows

    out = F.binary_cross_entropy_with_logits(
        mt.Tensor(logits), mt.Tensor(targets), mt.Tensor(pos_weight), reduction
    ).numpy()

    per_element = np.array(
        [
            [
                _bce_logits_ref(logits[i][j], targets[i][j], pos_weight[j])
                for j in range(3)
            ]
            for i in range(2)
        ]
    )
    expected = {
        "none": per_element,
        "sum": per_element.sum(),
        "mean": per_element.mean(),
    }[reduction]
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)


def test_bce_with_logits_pos_weight_gradcheck():
    logits = [[-2.0, 0.5, 3.0], [1.0, -1.5, 0.0]]
    targets = [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]
    pos_weight = [0.5, 2.0, 3.0]

    x = mt.Tensor(logits, requires_grad=True)
    F.binary_cross_entropy_with_logits(
        x, mt.Tensor(targets), mt.Tensor(pos_weight), "mean"
    ).backward()
    analytic = x.grad.numpy()
    mt.clear_autograd_graph()

    def loss_at(values):
        return F.binary_cross_entropy_with_logits(
            mt.Tensor(values), mt.Tensor(targets), mt.Tensor(pos_weight), "mean"
        ).numpy()

    h = 1e-3
    for i in range(2):
        for j in range(3):
            plus = [row[:] for row in logits]
            plus[i][j] += h
            minus = [row[:] for row in logits]
            minus[i][j] -= h
            central = (loss_at(plus) - loss_at(minus)) / (2 * h)
            # 1e-4: the difference of two f32 means cannot resolve finer.
            np.testing.assert_allclose(analytic[i][j], central, atol=1e-4)


def test_bce_with_logits_pos_weight_one_matches_unweighted():
    logits = mt.Tensor([[-2.0, 0.5], [3.0, -1.0]])
    targets = mt.Tensor([[1.0, 0.0], [1.0, 1.0]])
    plain = F.binary_cross_entropy_with_logits(logits, targets)
    weighted = F.binary_cross_entropy_with_logits(
        logits, targets, mt.Tensor([1.0, 1.0])
    )
    np.testing.assert_allclose(plain.numpy(), weighted.numpy(), rtol=1e-6)


def test_bce_with_logits_module_and_repr():
    logits = mt.Tensor([[0.4], [-0.7]])
    targets = mt.Tensor([[1.0], [0.0]])
    module = nn.BCEWithLogitsLoss()
    assert module.reduction == "mean"
    assert repr(module) == "BCEWithLogitsLoss(reduction='mean')"
    assert module.pos_weight is None
    np.testing.assert_allclose(
        module(logits, targets).numpy(),
        F.binary_cross_entropy_with_logits(logits, targets).numpy(),
        rtol=1e-6,
    )

    weighted = nn.BCEWithLogitsLoss("sum", mt.Tensor([2.0]))
    assert weighted.pos_weight is not None
    assert repr(weighted) == "BCEWithLogitsLoss(reduction='sum', pos_weight=...)"


def test_bce_with_logits_rejects_bad_inputs():
    logits = mt.Tensor([[0.4], [-0.7]])
    targets = mt.Tensor([[1.0], [0.0]])
    with pytest.raises(Exception):
        F.binary_cross_entropy_with_logits(logits, targets, reduction="median")
    with pytest.raises(Exception):
        F.binary_cross_entropy_with_logits(logits, mt.Tensor([[1.0, 0.0]]))
    with pytest.raises(Exception):
        # pos_weight of length 3 cannot broadcast against a width-1 logit
        F.binary_cross_entropy_with_logits(logits, targets, mt.Tensor([1.0, 2.0, 3.0]))


def test_conv2d_basic():
    x = Tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    w = Tensor([[[[1.0]]]])
    b = Tensor([1.0])
    y = F.conv2d(x, w, b)
    np.testing.assert_allclose(y.numpy(), np.array([[[[2.0, 3.0], [4.0, 5.0]]]]))


def test_conv2d_padding_stride():
    x = Tensor(np.arange(1, 17, dtype=np.float32).reshape(1, 1, 4, 4))
    w = Tensor(np.array([1, 0, 0, 1], dtype=np.float32).reshape(1, 1, 2, 2))
    y = F.conv2d(x, w, stride=2, padding=1)
    np.testing.assert_allclose(
        y.numpy(), np.array([[[[1.0, 3.0, 0.0], [9.0, 17.0, 8.0], [0.0, 14.0, 16.0]]]])
    )


def test_conv2d_kernel_too_large_raises():
    x = Tensor(np.zeros((1, 1, 2, 2), dtype=np.float32))
    w = Tensor(np.zeros((1, 1, 5, 5), dtype=np.float32))
    with pytest.raises(Exception):
        F.conv2d(x, w)


def test_conv2d_large_stride_output_shape():
    x = Tensor(np.arange(1, 17, dtype=np.float32).reshape(1, 1, 4, 4))
    w = Tensor(np.ones((1, 1, 1, 1), dtype=np.float32))
    y = F.conv2d(x, w, stride=5)
    assert y.shape == (1, 1, 1, 1)
    assert np.allclose(y.numpy()[0, 0, 0, 0], 1.0)


def test_one_hot_infers_classes_for_integer_tensor():
    labels = mt.Tensor([[0, 2], [1, 2]], dtype="int64")

    encoded = mt.one_hot(labels)

    expected = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    assert encoded.shape_vec() == [2, 2, 3]
    assert encoded.dtype == "float32"
    np.testing.assert_array_equal(encoded.numpy(), expected)


def test_one_hot_accepts_sequence_and_output_dtype():
    encoded = mt.functional.one_hot([2, 0, 1], num_classes=4, dtype="int64")

    expected = np.array(
        [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]],
        dtype=np.int64,
    )
    assert encoded.shape_vec() == [3, 4]
    assert encoded.dtype == "int64"
    np.testing.assert_array_equal(encoded.numpy(), expected)


def test_one_hot_supports_empty_input_with_explicit_classes():
    labels = mt.Tensor([], dtype="int64")

    encoded = mt.one_hot(labels, num_classes=3, dtype="bool")

    assert encoded.shape_vec() == [0, 3]
    assert encoded.dtype == "bool"
    np.testing.assert_array_equal(encoded.numpy(), np.empty((0, 3), dtype=bool))


def test_one_hot_preserves_nested_python_sequence_shape():
    encoded = mt.one_hot(((0, 1), (2, 1)), dtype="int32")

    expected = np.array(
        [
            [[1, 0, 0], [0, 1, 0]],
            [[0, 0, 1], [0, 1, 0]],
        ],
        dtype=np.int32,
    )
    assert encoded.shape_vec() == [2, 2, 3]
    assert encoded.dtype == "int32"
    np.testing.assert_array_equal(encoded.numpy(), expected)


def test_one_hot_accepts_numpy_integer_arrays_and_bool_sequences():
    encoded = mt.one_hot(np.array([1, 0], dtype=np.int32), num_classes=2)
    np.testing.assert_array_equal(
        encoded.numpy(), np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    )

    bool_encoded = mt.one_hot([True, False], num_classes=2, dtype="bool")
    np.testing.assert_array_equal(
        bool_encoded.numpy(), np.array([[False, True], [True, False]])
    )


def test_one_hot_rejects_invalid_labels_and_class_counts():
    with pytest.raises(ValueError, match="non-negative"):
        mt.one_hot(mt.Tensor([-1], dtype="int64"))

    with pytest.raises(ValueError, match="valid range"):
        mt.one_hot(mt.Tensor([3], dtype="int64"), num_classes=3)

    with pytest.raises(ValueError, match="must be provided"):
        mt.one_hot(mt.Tensor([], dtype="int64"))

    with pytest.raises(TypeError, match="integer or bool dtype"):
        mt.one_hot(mt.Tensor([0.0], dtype="float32"))

    with pytest.raises(TypeError, match="integer or bool dtype"):
        mt.one_hot(1.5)


def test_bincount_counts_bool_and_integer_labels():
    labels = mt.Tensor([0, 2, 1, 2, 2], dtype="int64")
    counts = mt.bincount(labels)
    assert counts.dtype == "int64"
    assert np.array_equal(counts.numpy(), np.array([1, 1, 3], dtype=np.int64))

    bool_counts = mt.functional.bincount([True, False, True], minlength=3)
    assert np.array_equal(bool_counts.numpy(), np.array([1, 2, 0], dtype=np.int64))


def test_bincount_weighted_and_empty_edge_cases():
    labels = mt.Tensor([0, 1, 1, 3], dtype="int32")
    weights = mt.Tensor([0.5, 1.25, 2.75, -1.0], dtype="float64")
    counts = mt.bincount(labels, weights=weights, minlength=5)
    expected = np.array([0.5, 4.0, 0.0, -1.0, 0.0], dtype=np.float64)
    assert counts.dtype == "float64"
    assert np.allclose(counts.numpy(), expected)

    f32_counts = mt.bincount(labels, weights=weights.astype("float32"))
    assert f32_counts.dtype == "float32"
    assert np.allclose(f32_counts.numpy(), expected[:4].astype(np.float32))

    empty = mt.bincount(mt.Tensor([], dtype="int64"), minlength=2)
    assert np.array_equal(empty.numpy(), np.array([0, 0], dtype=np.int64))


def test_bincount_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="non-negative"):
        mt.bincount(mt.Tensor([0, -1], dtype="int64"))
    with pytest.raises(ValueError, match="1-D"):
        mt.bincount(mt.Tensor(1, dtype="int64"))
    with pytest.raises(ValueError, match="1-D"):
        mt.bincount(mt.Tensor([[0, 1]], dtype="int64"))
    with pytest.raises(TypeError, match="integer or bool dtype"):
        mt.bincount(mt.Tensor([0.0], dtype="float32"))
    with pytest.raises(ValueError, match="same shape"):
        mt.bincount(mt.Tensor([0, 1], dtype="int64"), weights=mt.Tensor([1.0]))
    with pytest.raises(ValueError, match="same shape"):
        mt.bincount(mt.Tensor([0, 1], dtype="int64"), weights=mt.Tensor([[1.0, 2.0]]))
    with pytest.raises(TypeError, match="floating-point"):
        mt.bincount(
            mt.Tensor([0, 1], dtype="int64"), weights=mt.Tensor([1, 2], dtype="int64")
        )


def _softmax_np(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def test_focal_loss_forward_matches_definition():
    # mean reduction averages over samples (not numel): only the true-class term
    # is non-zero per sample. FL = alpha * (1 - p_t)^gamma * -log(p_t).
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((6, 4))
    cls = rng.integers(0, 4, 6)
    for alpha, gamma in [(0.25, 2.0), (0.5, 1.0), (0.75, 3.0)]:
        fl = nn.FocalLoss(alpha=alpha, gamma=gamma, reduction="mean")
        got = fl(mt.Tensor(logits), mt.Tensor(cls.tolist(), dtype="int64")).numpy()
        p = _softmax_np(logits)
        pt = p[np.arange(6), cls]
        ref = (alpha * (1 - pt) ** gamma * (-np.log(pt))).mean()
        np.testing.assert_allclose(got, ref, rtol=1e-5)


def test_focal_loss_gradient_matches_finite_diff():
    rng = np.random.default_rng(1)
    logits = rng.standard_normal((5, 3))
    cls = rng.integers(0, 3, 5)
    tgt = mt.Tensor(cls.tolist(), dtype="int64")
    alpha, gamma = 0.25, 2.0
    fl = nn.FocalLoss(alpha=alpha, gamma=gamma, reduction="mean")

    x = mt.Tensor(logits, requires_grad=True)
    fl(x, tgt).backward()
    grad = x.grad.numpy()

    def fwd(z):
        p = _softmax_np(z)
        pt = p[np.arange(5), cls]
        return (alpha * (1 - pt) ** gamma * (-np.log(pt))).mean()

    num = np.zeros_like(logits)
    eps = 1e-6
    for i in np.ndindex(*logits.shape):
        zp = logits.copy()
        zp[i] += eps
        zm = logits.copy()
        zm[i] -= eps
        num[i] = (fwd(zp) - fwd(zm)) / (2 * eps)
    np.testing.assert_allclose(grad, num, rtol=1e-3, atol=1e-6)


def _finite_diff_grad(np_loss, x0, eps=1e-6):
    num = np.zeros_like(x0)
    for i in np.ndindex(*x0.shape):
        xp = x0.copy()
        xp[i] += eps
        xm = x0.copy()
        xm[i] -= eps
        num[i] = (np_loss(xp) - np_loss(xm)) / (2 * eps)
    return num


def test_regression_losses_propagate_gradients():
    # MAE / Huber / SmoothL1 previously computed the loss on detached data, so
    # backward() failed (output had requires_grad=False). They must now flow
    # gradients matching finite differences.
    rng = np.random.default_rng(5)
    pred = rng.standard_normal((6, 3))
    targ = rng.standard_normal((6, 3))
    tt = mt.Tensor(targ)
    huber = lambda v: np.where(
        np.abs(v - targ) <= 1, 0.5 * (v - targ) ** 2, np.abs(v - targ) - 0.5
    )
    cases = [
        (lambda z: nn.MAELoss()(z, tt), lambda v: np.abs(v - targ).mean()),
        (lambda z: nn.HuberLoss()(z, tt), lambda v: huber(v).mean()),
        (lambda z: nn.SmoothL1Loss()(z, tt), lambda v: huber(v).mean()),
        (lambda z: F.smooth_l1_loss(z, tt, "mean"), lambda v: huber(v).mean()),
        (lambda z: F.smooth_l1_loss(z, tt, "sum"), lambda v: huber(v).sum()),
    ]
    for mt_loss, np_loss in cases:
        mt.clear_autograd_graph()
        x = mt.Tensor(pred, requires_grad=True)
        out = mt_loss(x)
        assert out.requires_grad
        out.backward()
        np.testing.assert_allclose(
            x.grad.numpy(), _finite_diff_grad(np_loss, pred), rtol=1e-3, atol=1e-6
        )


@pytest.mark.parametrize("negative_slope", [0.01, 0.1, 0.25, 1.0])
def test_leaky_relu_has_a_functional_form_and_matches_the_layer(negative_slope):
    # `relu`, `elu`, `selu`, `silu`, `gelu`, `softplus`, `hardshrink` and
    # `softsign` all had free-function forms; `leaky_relu` had only the
    # `nn.LeakyReLU` layer.
    values = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    tensor = mt.as_tensor(values)
    expected = np.where(values > 0, values, negative_slope * values)

    np.testing.assert_allclose(mt.leaky_relu(tensor, negative_slope).numpy(), expected)
    np.testing.assert_allclose(
        mt.functional.leaky_relu(tensor, negative_slope).numpy(), expected
    )
    np.testing.assert_allclose(tensor.leaky_relu(negative_slope).numpy(), expected)


def test_leaky_relu_default_slope_matches_the_layer():
    values = np.array([-2.0, 2.0])
    tensor = mt.as_tensor(values)
    np.testing.assert_allclose(
        mt.leaky_relu(tensor).numpy(), mt.nn.LeakyReLU()(tensor).numpy()
    )
    np.testing.assert_allclose(mt.leaky_relu(tensor).numpy(), [-0.02, 2.0])


@pytest.mark.parametrize("negative_slope", [0.01, 0.25, 1.0])
def test_leaky_relu_gradient_takes_the_slope_side_at_zero(negative_slope):
    # The kink at zero has no derivative; the convention is the negative side,
    # which is what PyTorch uses and what `relu` here has always used. This
    # kernel used `>= 0` and so returned 1 at zero, disagreeing with the `relu`
    # directly above it in the same file.
    values = np.array([-1.0, 0.0, 1.0])
    tensor = mt.Tensor(values.copy(), dtype="float64").requires_grad_(True)
    mt.sum(mt.leaky_relu(tensor, negative_slope)).backward()
    np.testing.assert_allclose(
        tensor.grad.numpy(), [negative_slope, negative_slope, 1.0]
    )


def test_relu_and_leaky_relu_agree_on_which_side_zero_belongs_to():
    relu_input = mt.Tensor(np.array([0.0]), dtype="float64").requires_grad_(True)
    mt.sum(mt.relu(relu_input)).backward()

    leaky_input = mt.Tensor(np.array([0.0]), dtype="float64").requires_grad_(True)
    mt.sum(mt.leaky_relu(leaky_input, 0.5)).backward()

    # relu is leaky_relu with slope 0, so their gradients at zero must relate
    # the same way: both take the negative branch.
    assert relu_input.grad.numpy()[0] == 0.0
    assert leaky_input.grad.numpy()[0] == 0.5


# Class-index targets take a gather path through cross_entropy rather than the
# dense one-hot one. The dense path forms `-sum(one_hot * log_p)` over the class
# axis -- a 4-million-element multiply and reduction to read out 4096 numbers --
# behind four more full-size passes (a zeros, an `eq`, a `masked_fill` and a
# `negate`) that exist only so `0 * -inf` cannot make a NaN where there is no
# target mass. None of that arises when only the target class is touched.
#
# The backward takes the indices directly too: `softmax - one_hot` needs no
# one-hot, since subtracting 1 at one column per row is a scatter.
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("dtype,index_dtype", [("float32", np.int64), ("float64", np.int32)])
def test_cross_entropy_index_targets_match_the_dense_formula(reduction, dtype, index_dtype):
    rng = np.random.default_rng(hash((reduction, dtype)) % 2**31)
    rows, classes = 64, 40
    logits = rng.standard_normal((rows, classes)).astype(dtype)
    idx = rng.integers(0, classes, rows).astype(index_dtype)

    x = mt.Tensor(logits, dtype=dtype, requires_grad=True)
    loss = mt.nn.cross_entropy(x, mt.from_numpy(idx), reduction=reduction)
    (loss.sum() if reduction == "none" else loss).backward()
    grad = x.grad.numpy()
    value = loss.numpy()
    mt.clear_autograd_graph()

    a = logits.astype(np.float64)
    z = a - a.max(-1, keepdims=True)
    log_p = z - np.log(np.exp(z).sum(-1, keepdims=True))
    per_sample = -log_p[np.arange(rows), idx]
    expected = {"mean": per_sample.mean(), "sum": per_sample.sum(), "none": per_sample}[reduction]
    np.testing.assert_allclose(value, expected, rtol=1e-5, atol=1e-6)

    probabilities = np.exp(log_p)
    expected_grad = probabilities.copy()
    expected_grad[np.arange(rows), idx] -= 1.0
    if reduction == "mean":
        expected_grad /= rows
    np.testing.assert_allclose(grad, expected_grad, rtol=1e-5, atol=1e-6)


def test_cross_entropy_index_and_one_hot_targets_agree():
    """The two paths must not have drifted apart -- same loss, same gradient."""
    rng = np.random.default_rng(77)
    rows, classes = 32, 12
    logits = rng.standard_normal((rows, classes)).astype(np.float32)
    idx = rng.integers(0, classes, rows)
    one_hot = np.zeros((rows, classes), dtype=np.float32)
    one_hot[np.arange(rows), idx] = 1.0

    results = []
    for target in (mt.from_numpy(idx.astype(np.int64)), mt.from_numpy(one_hot)):
        x = mt.Tensor(logits, dtype="float32", requires_grad=True)
        loss = mt.nn.cross_entropy(x, target)
        loss.backward()
        results.append((loss.numpy(), x.grad.numpy()))
        mt.clear_autograd_graph()

    np.testing.assert_allclose(results[0][0], results[1][0], rtol=1e-6)
    np.testing.assert_allclose(results[0][1], results[1][1], rtol=1e-5, atol=1e-7)


def test_cross_entropy_rejects_an_out_of_range_class_index():
    logits = mt.from_numpy(np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(Exception):
        mt.nn.cross_entropy(logits, mt.from_numpy(np.array([0, 5], dtype=np.int64)))
