# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Autograd coverage for indexing/shape/selection ops.

These operations previously produced tensors that reported ``requires_grad`` but
silently returned ``None`` gradients (no backward was wired). Each test compares
the analytic gradient against a float64 central finite difference so the
gradient math is checked end to end.
"""

import numpy as np
import pytest

import minitensor as mt


def _numeric_grad(fn, x_np, weights, eps=1e-6):
    grad = np.zeros_like(x_np)
    it = np.nditer(x_np, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        plus = x_np.copy()
        plus[idx] += eps
        minus = x_np.copy()
        minus[idx] -= eps
        grad[idx] = (fn(plus) - fn(minus)) / (2 * eps)
    return grad


def _analytic_grad(build, x_np, weights):
    x = mt.Tensor(x_np.tolist(), dtype="float64", requires_grad=True)
    out = build(x)
    if isinstance(out, tuple):
        out = out[0]
    (out * mt.Tensor(weights.tolist(), dtype="float64")).sum().backward()
    return x.grad.numpy()


def _out_shape(build, x_np):
    out = build(mt.Tensor(x_np.tolist(), dtype="float64"))
    if isinstance(out, tuple):
        out = out[0]
    return out.numpy().shape


def _check(build, shape=(3, 4), seed=0):
    rng = np.random.default_rng(seed)
    x_np = rng.standard_normal(shape)
    weights = rng.standard_normal(_out_shape(build, x_np))

    def scalar(x_perturbed):
        out = build(mt.Tensor(x_perturbed.tolist(), dtype="float64"))
        if isinstance(out, tuple):
            out = out[0]
        return float((out.numpy() * weights).sum())

    analytic = _analytic_grad(build, x_np, weights)
    numeric = _numeric_grad(scalar, x_np, weights)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-6)


def test_abs_grad():
    _check(lambda t: t.abs())


def test_clamp_grad():
    _check(lambda t: t.clamp(-0.3, 0.5))
    _check(lambda t: t.clamp_min(-0.2))
    _check(lambda t: t.clamp_max(0.4))


def test_basic_indexing_grad():
    _check(lambda t: t[:, 1:3])
    _check(lambda t: t[::2, 1:])
    _check(lambda t: t[1])
    _check(lambda t: t.narrow(1, 0, 2))


def test_index_select_and_gather_grad():
    _check(lambda t: t.index_select(1, [0, 2, 2, 1]))
    idx = mt.Tensor([[0, 1, 2, 3], [3, 2, 1, 0], [1, 1, 2, 2]], dtype="int64")
    _check(lambda t: t.gather(1, idx))


def test_flip_roll_repeat_grad():
    _check(lambda t: t.flip([0, 1]))
    _check(lambda t: t.roll(1, 1))
    _check(lambda t: t.roll(3))  # flattened, no-axis path
    _check(lambda t: t.roll([1, 2], [0, 1]))
    _check(lambda t: t.repeat(2, 2))
    _check(lambda t: t.repeat(3), shape=(4,))


def test_cat_stack_grad():
    _check(lambda t: mt.cat([t, t], 1))  # repeated input accumulates
    _check(lambda t: mt.cat([t, t * 2, t], 0))
    _check(lambda t: mt.stack([t, t], 0))


def test_min_max_grad():
    _check(lambda t: t.max())
    _check(lambda t: t.min())
    _check(lambda t: t.max(dim=1))
    _check(lambda t: t.min(dim=0))
    _check(lambda t: t.max(dim=1, keepdim=True))


def test_sort_topk_grad():
    _check(lambda t: t.sort(1))
    _check(lambda t: t.sort(0, descending=True))
    _check(lambda t: t.topk(2, 1))
    _check(lambda t: t.topk(3, 0))


def test_topk_forward_non_trailing_axis_matches_numpy():
    # Regression: topk on a non-last dim used to write output elements in the
    # wrong storage order for inner > 1.
    x_np = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    values = mt.Tensor(x_np.tolist(), dtype="float64").topk(2, dim=1)[0].numpy()
    expected = -np.sort(-x_np, axis=1)[:, :2, :]
    np.testing.assert_allclose(values, expected)


def test_median_and_nan_reduction_grad():
    # Odd length keeps the median/quantile bracketing unambiguous.
    _check(lambda t: t.median(), shape=(3, 5))
    _check(lambda t: t.median(dim=1), shape=(3, 5))
    _check(lambda t: t.nanmedian(), shape=(3, 5))
    _check(lambda t: t.nanmax(), shape=(3, 5))
    _check(lambda t: t.nanmin(dim=1), shape=(3, 5))


@pytest.mark.parametrize("q", [0.0, 0.3, 0.5, 1.0])
@pytest.mark.parametrize(
    "interpolation", ["linear", "lower", "higher", "midpoint", "nearest"]
)
def test_quantile_grad(q, interpolation):
    _check(lambda t: t.quantile(q, dim=1, interpolation=interpolation), shape=(3, 5))


def test_quantile_global_grad():
    _check(lambda t: t.quantile(0.4), shape=(3, 5))


def test_nanquantile_grad():
    _check(lambda t: t.nanquantile(0.5), shape=(3, 5))
    _check(lambda t: t.nanquantile(0.4, dim=1), shape=(3, 5))


def test_nanquantile_ignores_nan_in_gradient():
    x = mt.Tensor(
        [[1.0, float("nan"), 3.0, 5.0, 4.0]], dtype="float64", requires_grad=True
    )
    x.nanquantile(0.0).backward()  # min over the non-NaN values is 1.0
    np.testing.assert_allclose(x.grad.numpy(), [[1.0, 0.0, 0.0, 0.0, 0.0]])


def test_repeat_interleave_grad():
    _check(lambda t: t.repeat_interleave(2), shape=(3, 5))  # flattened path
    _check(lambda t: t.repeat_interleave(2, dim=1), shape=(3, 5))
    _check(
        lambda t: t.repeat_interleave(mt.Tensor([1, 2, 1, 2, 1], dtype="int64"), dim=1),
        shape=(3, 5),
    )


def test_nanmax_skips_nan_in_gradient():
    x = mt.Tensor([[1.0, float("nan"), 3.0, 2.0]], dtype="float64", requires_grad=True)
    x.nanmax().backward()
    np.testing.assert_allclose(x.grad.numpy(), [[0.0, 0.0, 1.0, 0.0]])


def test_max_global_ties_distribute_equally():
    x_np = np.array([[5.0, 5.0, 1.0], [2.0, 5.0, 3.0]])
    x = mt.Tensor(x_np.tolist(), dtype="float64", requires_grad=True)
    x.max().backward()
    mask = (x_np == 5.0).astype(float)
    np.testing.assert_allclose(x.grad.numpy(), mask / mask.sum())


@pytest.mark.parametrize(
    "build",
    [
        lambda t: t.unsqueeze(0),
        lambda t: t.unsqueeze(2),
        lambda t: t.unsqueeze(-1),
        lambda t: t.reshape(2, 1, 3, 4).squeeze(),
        lambda t: t.unsqueeze(1).squeeze(1),
        lambda t: t.flatten(),
        lambda t: t.flatten(1, 2),
        lambda t: t.ravel(),
        lambda t: mt.stack([t, t], 0),
    ],
    ids=[
        "unsqueeze0",
        "unsqueeze2",
        "unsqueeze-1",
        "squeeze_all",
        "squeeze_dim",
        "flatten",
        "flatten_range",
        "ravel",
        "stack",
    ],
)
def test_view_family_gradients_keep_input_shape(build):
    # These ops used to hand back a gradient with the *view's* shape (an extra or
    # missing size-1 axis) because they aliased the input's tensor id.
    x_np = np.random.default_rng(3).standard_normal((2, 3, 4))
    x = mt.Tensor(x_np.tolist(), dtype="float64", requires_grad=True)
    out = build(x)
    # How many times each input element appears in the output (stack uses x twice).
    multiplicity = out.numpy().size // x_np.size
    out.sum().backward()
    assert x.grad.numpy().shape == x_np.shape
    np.testing.assert_allclose(x.grad.numpy(), np.full(x_np.shape, float(multiplicity)))


@pytest.mark.parametrize("op", ["cat", "stack"])
def test_cat_stack_same_tensor_twice_accumulates(op):
    x_np = np.array([[1.0, 2.0], [3.0, 4.0]])
    x = mt.Tensor(x_np.tolist(), dtype="float64", requires_grad=True)
    if op == "cat":
        mt.cat([x, x], 0).sum().backward()
    else:
        mt.stack([x, x], 0).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), 2.0 * np.ones_like(x_np))
