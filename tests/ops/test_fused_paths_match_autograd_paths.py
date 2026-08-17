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
