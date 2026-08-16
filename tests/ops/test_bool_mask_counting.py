# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Summing a boolean mask counts its true entries.

`mask.sum()` is the ordinary way to ask how many elements satisfied a
condition, and it raised `Sum not supported for boolean tensors`. It was the
one hole in the boolean reductions -- `max`, `min`, `all`, `any`, `argmax`,
`argmin`, `sort` and `topk` all worked on masks already -- and the workaround
was to spell the cast out by hand:

    (x > 0).astype("int64").sum()

`bool` genuinely has no addition to accumulate in, which is what the rejection
was about, but that is an argument for widening the accumulator rather than for
refusing. The count goes into `int64`, and so does this now, with
`cumsum` following for the running count.

That widening is now the rule for every *accumulating* reduction rather than a
special case for masks: `sum`, `prod`, `cumsum` and `cumprod` all report
`int64` for a `bool` or `int32` input, which is what NumPy and PyTorch do. It
covers a real defect, not only a tidiness one -- summing three billion-ish
`int32` values used to return `1705032704`.

`max`, `min`, `argmax`, `argmin`, `sort` and `topk` are untouched: they report
a value that was already in the input, so there is nothing to widen. `mean`,
`var`, `std`, `norm` and `logsumexp` still raise on a mask, since what they
should return for one is a design question rather than an obvious one.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

MASK = np.array([[True, False, True], [True, True, False]])


def _mask(array):
    return mt.Tensor(array, dtype="bool")


def test_the_total_is_the_count():
    total = _mask(MASK).sum()
    assert total.item() == MASK.sum() == 4
    assert str(total.dtype) == "int64"


@pytest.mark.parametrize("dim", [0, 1, -1, -2])
@pytest.mark.parametrize("keepdim", [False, True])
def test_counting_along_a_dim_matches_numpy(dim, keepdim):
    got = _mask(MASK).sum(dim, keepdim)
    expected = MASK.sum(axis=dim, keepdims=keepdim)

    np.testing.assert_array_equal(got.numpy(), expected)
    assert tuple(got.shape_vec()) == expected.shape
    assert str(got.dtype) == "int64"


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_the_running_count_matches_numpy(dim):
    got = _mask(MASK).cumsum(dim)
    np.testing.assert_array_equal(got.numpy(), MASK.cumsum(axis=dim))
    assert str(got.dtype) == "int64"


def test_nansum_counts_too():
    """It delegates to `sum` for anything that cannot hold a NaN."""
    got = _mask(MASK).nansum()
    assert got.item() == 4
    assert str(got.dtype) == "int64"


def test_the_count_is_wider_than_the_mask():
    """The reason for `int64`: a count is not bounded by `bool`, and on a large
    mask a narrow accumulator would wrap rather than answer."""
    size = 3_000_000
    got = _mask(np.ones(size, dtype=bool)).sum()
    assert got.item() == size


@pytest.mark.parametrize(
    "shape,dim",
    [((0,), 0), ((0, 3), 0), ((2, 0), 1), ((0, 3), 1)],
)
def test_an_empty_mask_counts_zero(shape, dim):
    array = np.zeros(shape, dtype=bool)
    got = _mask(array).sum(dim)
    np.testing.assert_array_equal(got.numpy(), array.sum(axis=dim))


def test_it_agrees_with_the_cast_people_were_writing_by_hand():
    values = np.arange(-5, 7, dtype=np.float64).reshape(3, 4)
    tensor = mt.Tensor(values, dtype="float64")
    mask = tensor.gt(0.0)

    direct = mask.sum()
    spelled_out = mask.astype("int64").sum()

    assert direct.item() == spelled_out.item() == int((values > 0).sum())
    assert str(direct.dtype) == str(spelled_out.dtype)


def test_a_mask_from_a_comparison_counts_along_a_dim():
    """The shape this is actually used in: how many entries per row passed."""
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    got = mt.Tensor(values, dtype="float64").gt(5.0).sum(1)
    np.testing.assert_array_equal(got.numpy(), (values > 5.0).sum(axis=1))


# --- what deliberately did not change ---------------------------------------


@pytest.mark.parametrize("name", ["mean", "var", "std"])
def test_the_ambiguous_reductions_still_raise(name):
    """There is no obvious answer to adopt for these on a mask, so they are
    refused. The refusal names the operation."""
    with pytest.raises(Exception) as excinfo:
        getattr(_mask(MASK), name)()
    assert "boolean" in str(excinfo.value) or "float" in str(excinfo.value)


def test_prod_counts_in_int64_like_the_rest():
    """`prod` over a mask is `all` written differently, and it used to answer in
    `bool` while `sum` answered in `int64` -- the same operation family
    disagreeing with itself. Both widen now, and both match NumPy."""
    result = _mask(MASK).prod()
    assert str(result.dtype) == "int64"
    assert result.item() == int(MASK.prod())

    running = _mask(MASK).cumprod(0)
    assert str(running.dtype) == "int64"
    np.testing.assert_array_equal(running.numpy(), MASK.cumprod(axis=0))


@pytest.mark.parametrize("name", ["max", "min", "all", "any"])
def test_the_reductions_that_already_worked_are_unchanged(name):
    result = getattr(_mask(MASK), name)()
    assert str(result.dtype) == "bool"
    assert bool(result.item()) == bool(getattr(np, name)(MASK))


@pytest.mark.parametrize(
    "dtype,accumulated",
    [
        ("int32", "int64"),
        ("int64", "int64"),
        ("float32", "float32"),
        ("float64", "float64"),
    ],
)
def test_accumulating_reductions_report_the_accumulator_dtype(dtype, accumulated):
    """Narrow integers widen; floats do not. Promoting `float32` to `float64`
    would change every existing result and double the memory of the most common
    reduction in the library, and NumPy does not do it either."""
    array = np.arange(6).astype(dtype).reshape(2, 3)
    tensor = mt.Tensor(array, dtype=dtype)

    for got in (tensor.sum(), tensor.sum(0), tensor.cumsum(0), tensor.prod()):
        assert str(got.dtype) == accumulated
    np.testing.assert_array_equal(tensor.sum(1).numpy(), array.sum(axis=1))
    # ...and the dtype is the one NumPy reports for the same call.
    assert str(tensor.sum().dtype) == array.sum().dtype.name
