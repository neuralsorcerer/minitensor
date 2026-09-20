# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The NaN-skipping statistics that are arrangements, not kernels.

`nanprod`, `nanvar`, `nanstd`, `nanargmax` and `nanargmin` are each written in
terms of operations that already exist -- a product over a NaN-to-one
substitution, a mean of squared deviations from `nanmean`, an index reduction
over a tensor with NaN pushed to one end. That is one definition rather than
two, so what these tests have to establish is that the arrangement agrees with
NumPy on every value, and that the gradients the arrangement inherits are the
right ones rather than an accident of where the NaN went.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import minitensor as mt

# A grid with a different NaN pattern in every row: none, one, two, and (in the
# last row) all but one -- which is the boundary where an unbiased variance
# stops having a divisor.
VALUES = np.array(
    [
        [1.5, -2.25, 0.75, 4.0],
        [np.nan, 3.5, -1.5, 2.0],
        [0.5, np.nan, np.nan, -3.25],
        [np.nan, np.nan, 6.5, np.nan],
    ]
)


def _t(values, requires_grad=False):
    return mt.Tensor(
        np.array(values, dtype=np.float64), dtype="float64", requires_grad=requires_grad
    )


@pytest.mark.parametrize("keepdim", [False, True])
def test_nanprod_matches_numpy_along_an_axis(keepdim):
    got = _t(VALUES).nanprod(1, keepdim).numpy()
    want = np.nanprod(VALUES, axis=1, keepdims=keepdim)
    np.testing.assert_allclose(got, want, rtol=1e-15)


def test_nanprod_over_everything_matches_numpy():
    assert _t(VALUES).nanprod().item() == pytest.approx(np.nanprod(VALUES), rel=1e-15)


def test_an_all_nan_slice_has_a_product_of_one():
    # NaN reads as the multiplicative identity, so a slice of nothing but NaN
    # is an empty product -- which is 1, exactly as a genuinely empty one is.
    all_nan = np.array([[np.nan, np.nan], [2.0, 3.0]])
    np.testing.assert_array_equal(
        _t(all_nan).nanprod(1).numpy(), np.nanprod(all_nan, axis=1)
    )
    assert _t(all_nan).nanprod(1).numpy()[0] == 1.0


def test_nanprod_leaves_an_integer_tensor_alone():
    # An integer tensor holds no NaN, so `nanprod` is `prod` and has to stay in
    # the accumulating dtype rather than being routed through a float.
    integers = mt.Tensor(np.array([[2, 3], [4, 5]], dtype=np.int64), dtype="int64")
    np.testing.assert_array_equal(integers.nanprod(1).numpy(), np.array([6, 20]))
    assert "int64" in str(integers.nanprod(1).dtype)


@pytest.mark.parametrize("unbiased", [True, False])
@pytest.mark.parametrize("keepdim", [False, True])
def test_nanvar_and_nanstd_match_numpy(unbiased, keepdim):
    rows = VALUES[:3]  # the all-but-one row has no unbiased variance; below.
    ddof = 1 if unbiased else 0

    got_var = _t(rows).nanvar(1, unbiased, keepdim).numpy()
    want_var = np.nanvar(rows, axis=1, ddof=ddof, keepdims=keepdim)
    np.testing.assert_allclose(got_var, want_var, rtol=1e-13)

    got_std = _t(rows).nanstd(1, unbiased, keepdim).numpy()
    want_std = np.nanstd(rows, axis=1, ddof=ddof, keepdims=keepdim)
    np.testing.assert_allclose(got_std, want_std, rtol=1e-13)


def test_nanvar_over_everything_matches_numpy():
    for unbiased, ddof in ((True, 1), (False, 0)):
        got = _t(VALUES).nanvar(None, unbiased).item()
        assert got == pytest.approx(np.nanvar(VALUES, ddof=ddof), rel=1e-13)


def test_a_slice_without_enough_entries_reports_nan_rather_than_a_number():
    # One finite entry has no unbiased variance: the divisor is `1 - 1`. NumPy
    # answers NaN (with a warning); answering 0 instead would claim the slice
    # was measured and found not to vary.
    single = np.array([[np.nan, 4.0, np.nan]])
    assert np.isnan(_t(single).nanvar(1, True).numpy()[0])
    assert _t(single).nanvar(1, False).numpy()[0] == 0.0


@pytest.mark.parametrize("unbiased", [True, False])
def test_an_all_nan_slice_has_no_variance(unbiased):
    """With the correction applied -- which is the default -- this answered
    `-0.0`: the divisor is the non-NaN count less one, so a slice with nothing
    in it divided by *minus one* and came back as a clean zero. A slice of no
    data does not have zero variance; it has none, and anything normalising by
    it would have divided by that zero without a word.

    Clamping the divisor at zero makes the division `0 / 0`, which is NaN --
    what `var` answers for an empty slice and what NumPy answers for this one.
    """
    for slice_of_nothing in (np.array([[np.nan, np.nan]]), np.zeros((1, 0))):
        assert np.isnan(_t(slice_of_nothing).nanvar(1, unbiased).numpy()[0])
        assert np.isnan(_t(slice_of_nothing).nanstd(1, unbiased).numpy()[0])

    # One finite entry is the neighbouring case, and it was already right:
    # the correction takes the divisor to zero rather than past it.
    single = _t(np.array([[np.nan, 4.0, np.nan]]))
    assert np.isnan(single.nanvar(1, True).numpy()[0])
    assert single.nanvar(1, False).numpy()[0] == 0.0


@pytest.mark.parametrize(
    "name,ours,theirs",
    [
        ("nanprod", lambda t: mt.nanprod(t), np.nanprod),
        ("nancumsum", lambda t: mt.nancumsum(t), np.nancumsum),
        ("nansum", lambda t: mt.nansum(t), np.nansum),
        ("nanmax", lambda t: mt.nanmax(t), np.nanmax),
        ("nanmin", lambda t: mt.nanmin(t), np.nanmin),
    ],
)
def test_only_the_nan_is_replaced(name, ours, theirs):
    """`nanprod` and `nancumsum` filled the NaN positions with `nan_to_num`,
    whose defaults also replace the infinities -- with the dtype's finite
    extremes, which is what that function is for. So `nancumsum([inf, -inf])`
    added 1.8e308 to its negation and answered zero, where the running total
    is `inf` and then NaN, and `nanprod([nan, inf])` came back as a finite
    1.8e308 standing where an infinity was.

    Replacing only the NaN is what `nancumprod` already did, and it is the
    difference between skipping a value and clamping one.
    """
    values = np.array([np.nan, np.inf, 2.0, -np.inf, np.nan, 3.0])
    with np.errstate(invalid="ignore"):
        want = theirs(values)
    got = ours(_t(values)).numpy()
    np.testing.assert_array_equal(got, want)


def test_an_infinity_survives_a_nan_skipping_product_along_an_axis():
    grid = np.array([[1.5, np.nan, -0.5], [2.0, np.inf, -np.inf]])
    np.testing.assert_array_equal(
        mt.nanprod(_t(grid), 0).numpy(), np.nanprod(grid, axis=0)
    )
    np.testing.assert_array_equal(
        mt.nancumprod(_t(grid), 1).numpy(), np.nancumprod(grid, axis=1)
    )


def test_nanvar_of_integers_is_var():
    """An integer holds no NaN, so there is nothing to skip and this is `var`
    -- the delegation `nanprod` and `nanmax` already make. It widens to
    `float64` as `var` does, rather than refusing the call."""
    integers = mt.Tensor(np.array([1, 2, 3], dtype=np.int64), dtype="int64")
    assert str(integers.nanvar().dtype) == "float64"
    np.testing.assert_allclose(
        integers.nanvar().item(), np.var([1.0, 2.0, 3.0], ddof=1)
    )
    np.testing.assert_allclose(integers.nanvar().item(), integers.var().item())


@pytest.mark.parametrize("unbiased", [False, True])
@pytest.mark.parametrize("keepdim", [False, True])
def test_nanvar_and_nanstd_over_several_axes_match_numpy(unbiased, keepdim):
    """These used to refuse a list of axes, because the count they divide by
    came from a `count_nonzero` that reduced one axis at a time -- so the
    refusal was about the divisor, not about the statistic. `count_nonzero`
    counts over as many axes as it is given now, and the divisor is right, so
    there is nothing left to refuse. `var` and `std` always took a list."""
    grid = np.stack([VALUES, VALUES[::-1] * 1.5])
    for axes in ([0, 1], [1, 2], [0, 2], [0, 1, 2], [-3, -1]):
        correction = 1 if unbiased else 0
        got_var = _t(grid).nanvar(axes, unbiased, keepdim).numpy()
        got_std = _t(grid).nanstd(axes, unbiased, keepdim).numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            want_var = np.nanvar(
                grid, axis=tuple(axes), ddof=correction, keepdims=keepdim
            )
            want_std = np.nanstd(
                grid, axis=tuple(axes), ddof=correction, keepdims=keepdim
            )
        np.testing.assert_allclose(got_var, want_var, rtol=1e-12, equal_nan=True)
        np.testing.assert_allclose(got_std, want_std, rtol=1e-12, equal_nan=True)


@pytest.mark.parametrize("keepdim", [False, True])
def test_nanargmax_and_nanargmin_match_numpy(keepdim):
    rows = VALUES[:3]
    got_max = _t(rows).nanargmax(1, keepdim).numpy()
    want_max = np.nanargmax(rows, axis=1)
    np.testing.assert_array_equal(
        got_max, want_max.reshape(-1, 1) if keepdim else want_max
    )

    got_min = _t(rows).nanargmin(1, keepdim).numpy()
    want_min = np.nanargmin(rows, axis=1)
    np.testing.assert_array_equal(
        got_min, want_min.reshape(-1, 1) if keepdim else want_min
    )


def test_nanargmax_over_everything_matches_numpy():
    assert _t(VALUES).nanargmax().item() == int(np.nanargmax(VALUES))
    assert _t(VALUES).nanargmin().item() == int(np.nanargmin(VALUES))


def test_the_index_points_past_a_leading_nan():
    # The whole point: the largest entry sits behind a NaN, and a plain
    # `argmax` would stop at the NaN because no comparison against it is true.
    values = np.array([np.nan, 1.0, 9.0, 2.0])
    assert _t(values).nanargmax().item() == 2
    assert _t(values).nanargmin().item() == 1


def test_an_all_nan_slice_has_no_index_to_report():
    # NumPy raises here, and so does this: every index it could return points
    # at a NaN, so there is no answer to give.
    with pytest.raises(Exception, match="no index"):
        _t(np.array([np.nan, np.nan])).nanargmax()
    one_empty_row = np.array([[1.0, 2.0], [np.nan, np.nan]])
    with pytest.raises(Exception, match="no index"):
        _t(one_empty_row).nanargmin(1)


def test_nanargmax_on_an_integer_tensor_is_argmax():
    integers = mt.Tensor(np.array([3, 9, 1], dtype=np.int64), dtype="int64")
    assert integers.nanargmax().item() == 1
    assert integers.nanargmin().item() == 2


def test_the_index_reductions_return_int64():
    assert "int64" in str(_t(VALUES[:3]).nanargmax(1).dtype)
    assert "int64" in str(_t(VALUES[:3]).nanargmin(1).dtype)


def test_the_functional_spellings_agree_with_the_methods():
    rows = VALUES[:3]
    for name, args in (
        ("nanprod", (1,)),
        ("nanvar", (1,)),
        ("nanstd", (1,)),
        ("nanargmax", (1,)),
        ("nanargmin", (1,)),
    ):
        tensor = _t(rows)
        np.testing.assert_array_equal(
            getattr(mt, name)(tensor, *args).numpy(),
            getattr(tensor, name)(*args).numpy(),
        )


def test_the_variance_gradient_reaches_every_finite_entry():
    # The deviation has to be zeroed at the NaN positions *before* it is
    # squared. Squaring first and dropping the NaN afterwards gives the same
    # total, but the chain rule then computes `0 * 2 * NaN` for the skipped
    # entry, and that NaN travels back through the shared mean into every
    # finite entry of the slice.
    values = np.array([1.5, np.nan, -2.0, 4.25])
    tensor = _t(values, requires_grad=True)
    tensor.nanvar(None, False).backward()
    got = tensor.grad.numpy()

    finite = ~np.isnan(values)
    count = finite.sum()
    want = np.zeros_like(values)
    want[finite] = 2.0 * (values[finite] - values[finite].mean()) / count

    assert not np.isnan(got[finite]).any(), "a skipped entry poisoned the gradient"
    np.testing.assert_allclose(got[finite], want[finite], rtol=1e-12)
    assert got[~finite] == 0.0


def test_the_product_gradient_treats_a_skipped_entry_as_a_one():
    values = np.array([2.0, np.nan, 3.0])
    tensor = _t(values, requires_grad=True)
    tensor.nanprod().backward()
    got = tensor.grad.numpy()

    # d(2 * 1 * 3)/dx is the product of the others, with the NaN reading as 1.
    np.testing.assert_allclose(got[0], 3.0, rtol=1e-14)
    np.testing.assert_allclose(got[2], 2.0, rtol=1e-14)
    assert not np.isnan(got[0]) and not np.isnan(got[2])


def test_nanstd_is_the_square_root_of_nanvar():
    rows = VALUES[:3]
    np.testing.assert_allclose(
        _t(rows).nanstd(1).numpy(), np.sqrt(_t(rows).nanvar(1).numpy()), rtol=1e-15
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "values",
    [
        [-np.inf, -np.inf],
        [np.inf, np.inf],
        [-np.inf, np.nan, -np.inf],
        [np.inf, np.nan, np.inf],
        [np.nan, np.nan, np.nan],
        [np.nan] * 40_000,
        [-np.inf] + [np.nan] * 40_000,
    ],
)
def test_an_infinite_extremum_is_not_mistaken_for_the_seed(values, dtype):
    """The scan starts at `-inf` for a maximum, which is a value the data can
    hold. What tells the two apart is a separate flag saying whether anything
    that was not a NaN was seen at all -- so an all-NaN slice answers NaN while
    a slice of nothing but `-inf` answers `-inf`, and neither borrows the
    other's answer.

    Long enough in two cases to be folded by several threads, because the seed
    is per chunk: a chunk that saw only NaN has to hand back "nothing here"
    rather than its seed, or the merge would take the seed for a real value.
    """

    data = np.array(values, dtype=dtype)
    with warnings.catch_warnings():
        # NumPy warns on an all-NaN slice and still answers NaN, which is the
        # answer being checked.
        warnings.simplefilter("ignore", RuntimeWarning)
        for ours, theirs in ((mt.nanmax, np.nanmax), (mt.nanmin, np.nanmin)):
            want = theirs(data)
            got = ours(_t(data)).numpy()
            if np.isnan(want):
                assert np.isnan(got), (ours.__name__, values[:3], dtype)
            else:
                assert got == want, (ours.__name__, values[:3], dtype)
