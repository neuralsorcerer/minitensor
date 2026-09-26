# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`var`, `std` and `logsumexp` must not answer differently when not training.

Each has a fused single-axis kernel taken only when the tensor does *not*
require gradients; with gradients they go through a composition of primitive
ops instead. Two implementations of one operation, and which you get depends on
a flag that has nothing to do with the arithmetic.

They had drifted. The fused kernels walked the reduced axis with a plain
running total, so their error grew with the axis length while the composed path
-- built on `sum`, which folds its partials pairwise -- stayed flat. On a
4M-element axis `var` measured 3.8e-3 against the composed path's 1.2e-7, a
factor of 38000, and `logsumexp` 3.9e-4 against 3.7e-6. The bad answer was the
one you got when you were *not* training, which is also when you are most
likely to be reporting the number rather than differentiating it.

Both layouts were affected: the contiguous one where the reduced axis is last,
and the slab one where it is not.

These are equality-of-accuracy tests. The point is not that either path hits a
particular tolerance -- it is that the same call cannot mean two things.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

# Shapes chosen so the reduced axis is long enough for the growth to show, in
# both layouts: `inner == 1` when the axis is last, `inner > 1` when it is not.
LONG_AXIS_CASES = [
    ((4, 500_000), 1),  # contiguous rows
    ((500_000, 4), 0),  # slab, narrow
    ((2, 1_000_000), 1),
    ((1_000_000, 2), 0),
]


def _data(shape, seed=5):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) * 3).astype(np.float32)


def _relative(got, exact):
    return float(
        np.abs(np.asarray(got, np.float64) - exact).max() / np.abs(exact).max()
    )


def _logsumexp_reference(wide, axis):
    peak = wide.max(axis=axis, keepdims=True)
    return np.log(np.exp(wide - peak).sum(axis=axis)) + peak.squeeze(axis)


@pytest.mark.parametrize("shape,dim", LONG_AXIS_CASES)
@pytest.mark.parametrize("op", ["var", "std", "logsumexp"])
def test_the_fused_and_composed_paths_agree_on_a_long_axis(shape, dim, op):
    values = _data(shape)
    wide = values.astype(np.float64)
    if op == "logsumexp":
        exact = _logsumexp_reference(wide, dim)
        call = lambda t: t.logsumexp(dim)  # noqa: E731
    elif op == "var":
        exact = wide.var(axis=dim, ddof=1)
        call = lambda t: t.var(dim=dim)  # noqa: E731
    else:
        exact = wide.std(axis=dim, ddof=1)
        call = lambda t: t.std(dim=dim)  # noqa: E731

    fused = _relative(call(mt.Tensor(values, dtype="float32")).numpy(), exact)
    composed = _relative(
        call(mt.Tensor(values, dtype="float32", requires_grad=True)).numpy(), exact
    )

    # Neither may be wildly worse than the other. The bound is loose because
    # they are genuinely different orderings; what it rules out is the
    # order-of-magnitude drift that was there.
    assert fused < 100 * max(
        composed, 1e-9
    ), f"fused {fused:.3e} vs composed {composed:.3e}"
    assert composed < 100 * max(
        fused, 1e-9
    ), f"composed {composed:.3e} vs fused {fused:.3e}"


@pytest.mark.parametrize("shape,dim", LONG_AXIS_CASES)
@pytest.mark.parametrize("op", ["var", "std", "logsumexp"])
def test_neither_path_degrades_with_axis_length(shape, dim, op):
    """The property the blocking exists for: error must not track `n`.

    float32 carries about seven digits, so anything under 1e-5 on a
    million-element axis means the accumulation is not a running total."""
    values = _data(shape, seed=9)
    wide = values.astype(np.float64)
    if op == "logsumexp":
        exact = _logsumexp_reference(wide, dim)
        call = lambda t: t.logsumexp(dim)  # noqa: E731
    elif op == "var":
        exact = wide.var(axis=dim, ddof=1)
        call = lambda t: t.var(dim=dim)  # noqa: E731
    else:
        exact = wide.std(axis=dim, ddof=1)
        call = lambda t: t.std(dim=dim)  # noqa: E731

    for requires_grad in (False, True):
        got = call(mt.Tensor(values, dtype="float32", requires_grad=requires_grad))
        error = _relative(got.numpy(), exact)
        assert error < 1e-5, f"requires_grad={requires_grad} error {error:.3e}"


@pytest.mark.parametrize("shape,dim", [((3, 7), 1), ((7, 3), 0), ((2, 3, 5), 2)])
@pytest.mark.parametrize("op", ["var", "std", "logsumexp"])
def test_small_shapes_are_unchanged(shape, dim, op):
    """The blocking must not reach short axes, which take the direct loop."""
    values = _data(shape, seed=3)
    wide = values.astype(np.float64)
    t = mt.Tensor(values, dtype="float32")
    if op == "logsumexp":
        want = _logsumexp_reference(wide, dim)
        got = t.logsumexp(dim)
    elif op == "var":
        want = wide.var(axis=dim, ddof=1)
        got = t.var(dim=dim)
    else:
        want = wide.std(axis=dim, ddof=1)
        got = t.std(dim=dim)
    np.testing.assert_allclose(got.numpy(), want, rtol=1e-5, atol=1e-6)


def test_var_still_matches_numpy_on_the_biased_convention():
    """`var` is unbiased by default, like PyTorch; NumPy's default is biased.
    The blocking must not have quietly moved which one it computes."""
    values = _data((4, 1000), seed=11)
    t = mt.Tensor(values, dtype="float32")
    np.testing.assert_allclose(
        t.var(dim=1).numpy(),
        values.astype(np.float64).var(axis=1, ddof=1),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        t.var(dim=1, unbiased=False).numpy(),
        values.astype(np.float64).var(axis=1),
        rtol=1e-5,
    )


def test_logsumexp_still_reports_the_max_for_non_finite_rows():
    """The fused path reproduces the composed path's limit for a row whose max
    is not finite; blocking the sum must not have lost it."""
    values = np.array(
        [[np.inf, 1.0, 2.0], [-np.inf, -np.inf, -np.inf], [np.nan, 1.0, 2.0]],
        dtype=np.float64,
    )
    got = mt.Tensor(values, dtype="float64").logsumexp(1).numpy()
    assert got[0] == np.inf
    assert got[1] == -np.inf
    assert np.isnan(got[2])


# One case per route the fused `logsumexp` can take: a long contiguous run
# spread across the pool, many short runs, a slab spread across the pool, and
# many slabs, narrow and wide.
LOGSUMEXP_ROUTES = [
    ((300_000,), 0),
    ((64, 4096), 1),
    ((4096, 64), 0),
    ((16, 3000, 3), 1),
    ((6, 300, 200), 1),
]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("shape,dim", LOGSUMEXP_ROUTES)
def test_logsumexp_non_finite_values_mean_the_same_on_both_paths(shape, dim, dtype):
    # A NaN anywhere in a slice makes it NaN, a `+inf` makes it `+inf`, and a
    # slice that is all `-inf` is `-inf`; everything else is finite. The fused
    # kernel records a NaN beside each lane rather than branching on it, so
    # this pins that the record reaches the answer on every route.
    rng = np.random.default_rng(len(shape) * 7 + dim)
    values = rng.standard_normal(shape).astype(dtype)
    moved = np.moveaxis(values, dim, -1).reshape(-1, shape[dim])
    rows = moved.shape[0]
    moved[0, rng.integers(shape[dim])] = np.nan
    moved[1 % rows, rng.integers(shape[dim])] = np.inf
    moved[2 % rows, :] = -np.inf
    moved[3 % rows, rng.integers(shape[dim])] = -np.inf
    values = np.moveaxis(moved.reshape(np.moveaxis(values, dim, -1).shape), -1, dim)
    values = np.ascontiguousarray(values)

    fused = mt.Tensor(values, dtype=dtype).logsumexp(dim).numpy()
    composed = (
        mt.Tensor(values, dtype=dtype, requires_grad=True)
        .logsumexp(dim)
        .detach()
        .numpy()
    )
    np.testing.assert_array_equal(np.isnan(fused), np.isnan(composed))
    np.testing.assert_array_equal(np.isposinf(fused), np.isposinf(composed))
    np.testing.assert_array_equal(np.isneginf(fused), np.isneginf(composed))
    finite = np.isfinite(composed)
    atol = 1e-5 if dtype == "float32" else 1e-12
    np.testing.assert_allclose(fused[finite], composed[finite], rtol=atol, atol=atol)


# One shape per route the fused `var` takes down a non-last axis.
VAR_ROUTES = [
    ((512, 4096), 0),  # one short slab, split by columns
    ((4096, 1024), 0),  # one tall slab, split by rows
    ((100_000, 8), 0),  # one tall narrow slab
    ((2, 300_000, 5), 1),  # few slabs, each split by rows
    ((16, 100_000, 2), 1),  # many narrow slabs
    ((8, 512, 512), 1),  # many wide slabs
]


def _ulps(got, exact):
    nearest = exact.astype(got.dtype)
    kind = np.int32 if got.dtype == np.float32 else np.int64
    return int(
        np.abs(
            got.view(kind).astype(np.int64) - nearest.view(kind).astype(np.int64)
        ).max()
    )


@pytest.mark.parametrize("shape,dim", VAR_ROUTES)
def test_var_down_a_non_last_axis_stays_within_a_few_ulps(shape, dim):
    """The fused `var` accumulated each column of a slab in one chain of up to
    8192 rows: 40 ulps from the exact answer on a `(4096, 1024)` matrix down
    its rows, 23 on `(64, 20000, 3)` down its middle axis. It now folds through
    the same short blocked chains `sum` does."""
    rng = np.random.default_rng(len(shape) * 100 + dim)
    values = (rng.random(shape) + 0.5).astype(np.float32)
    exact = values.astype(np.float64).var(axis=dim, ddof=1)
    t = mt.from_numpy(values)
    assert _ulps(t.var(dim).numpy(), exact) <= 8
    assert _ulps(t.std(dim).numpy(), np.sqrt(exact)) <= 8
