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


def test_an_all_nan_slice_has_no_variance():
    assert np.isnan(_t(np.array([[np.nan, np.nan]])).nanvar(1, False).numpy()[0])


def test_nanvar_rejects_a_non_float_tensor():
    integers = mt.Tensor(np.array([1, 2, 3], dtype=np.int64), dtype="int64")
    with pytest.raises(Exception, match="floating point"):
        integers.nanvar()


def test_reducing_more_than_one_dimension_at_a_time_is_refused():
    # The non-NaN count comes from a single-axis `count_nonzero`, so two axes
    # would divide by the wrong count. Saying so beats a quietly wrong answer.
    with pytest.raises(Exception, match="one dimension at a time"):
        _t(VALUES).nanvar([0, 1])


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
