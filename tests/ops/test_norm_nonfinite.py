# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The norm of a slice holding an infinity is infinite, not NaN.

`norm` computes `m * (sum |x/m|^p)^(1/p)` with `m = max|x|` rather than the
direct `(sum |x|^p)^(1/p)`, because squaring first overflows f32 once `|x|`
passes about 1.8e19 -- and reporting `inf` for a norm that is perfectly
representable defeats the point of calling `norm` to detect a blow-up.

The scaling had the same failure one step further out. When the slice already
contains an infinity, `m` is infinite, and dividing by it gives `inf / inf` for
that element and `0` for every other. So the norm came back NaN:

    mt.Tensor([1.0, inf, 2.0]).norm(2)   ->  nan
    np.linalg.norm([1.0, inf, 2.0])      ->  inf

for every finite `p > 0`, in both float dtypes, whole-tensor and along a dim.
`p = 0` and `p = +/-inf` were right, because those orders never divide by the
scale. NaN and infinity are different diagnoses -- one says something computed
`0 * inf` or `inf - inf`, the other says a magnitude overflowed -- so turning
the second into the first loses exactly the information the call was for.

There is nothing to scale by when the maximum magnitude is zero or infinite, so
one now stands in for it: the accumulation runs unscaled, `inf` survives it, and
multiplying by one at the end keeps it. A NaN anywhere in the slice still
poisons the sum and wins over an infinity, which is what NumPy does too.

The tests check against NumPy rather than against a hand-written expectation,
and they keep the overflow guard pinned in the same file, because the two pull
in opposite directions: the guard is why the scaling exists, and this is the
case the scaling got wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

DTYPES = ["float32", "float64"]

# Every order the scaling path handles, plus the three that bypass it.
SCALED_ORDERS = [0.5, 1.0, 1.5, 2.0, 3.0]
UNSCALED_ORDERS = [0.0, float("inf"), float("-inf")]

NONFINITE_VECTORS = {
    "inf": [1.0, np.inf, 2.0],
    "neg_inf": [1.0, -np.inf, 2.0],
    "all_inf": [np.inf, np.inf],
    "inf_and_nan": [np.inf, np.nan],
    "nan": [1.0, np.nan],
    "nan_first": [np.nan, 1.0],
}


def _agree(got, expected):
    """NaN equals NaN here; `assert_allclose` already treats infinities that
    way but not NaN against a finite value."""
    got, expected = float(got), float(expected)
    if np.isnan(expected):
        assert np.isnan(got), f"expected nan, got {got}"
    elif np.isinf(expected):
        assert np.isinf(got) and np.sign(got) == np.sign(
            expected
        ), f"expected {expected}, got {got}"
    else:
        np.testing.assert_allclose(got, expected, rtol=1e-5)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("p", SCALED_ORDERS + UNSCALED_ORDERS)
@pytest.mark.parametrize(
    "name,values", NONFINITE_VECTORS.items(), ids=list(NONFINITE_VECTORS)
)
def test_nonfinite_norms_match_numpy(name, values, p, dtype):
    array = np.array(values, dtype=dtype)
    got = mt.Tensor(array, dtype=dtype).norm(p).numpy()
    _agree(got, np.linalg.norm(array.astype(np.float64), ord=p))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("p", SCALED_ORDERS)
def test_an_infinity_gives_an_infinite_norm_not_nan(p, dtype):
    """The regression itself, stated directly."""
    got = mt.Tensor(np.array([1.0, np.inf, 2.0], dtype=dtype), dtype=dtype).norm(p)
    assert np.isinf(got.numpy()), f"p={p} gave {got.numpy()}"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("p", SCALED_ORDERS)
def test_a_negative_infinity_counts_by_its_magnitude(p, dtype):
    """`|-inf|` is `inf`, so the sign of the offending element must not reach
    the answer."""
    plus = mt.Tensor(np.array([1.0, np.inf, 2.0], dtype=dtype), dtype=dtype).norm(p)
    minus = mt.Tensor(np.array([1.0, -np.inf, 2.0], dtype=dtype), dtype=dtype).norm(p)
    assert np.isinf(plus.numpy()) and np.isinf(minus.numpy())
    assert plus.numpy() > 0 and minus.numpy() > 0


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("p", SCALED_ORDERS)
def test_nan_still_wins_over_an_infinity(p, dtype):
    """Both are non-finite and they do not mean the same thing; NumPy resolves
    a slice holding both to NaN."""
    array = np.array([np.inf, np.nan], dtype=dtype)
    got = mt.Tensor(array, dtype=dtype).norm(p).numpy()
    assert np.isnan(got)
    _agree(got, np.linalg.norm(array.astype(np.float64), ord=p))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("p", SCALED_ORDERS)
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_only_the_affected_slice_goes_infinite(p, dim, dtype):
    """A reduction along a dim must not let one slice's infinity leak into the
    others -- the scale is per slice, and so is the substitution."""
    array = np.array([[1.0, np.inf], [3.0, 4.0]], dtype=dtype)

    got = mt.Tensor(array, dtype=dtype).norm(p, dim).numpy()

    expected = np.linalg.norm(array.astype(np.float64), ord=p, axis=dim)
    assert got.shape == expected.shape
    for a, b in zip(np.ravel(got), np.ravel(expected)):
        _agree(a, b)


@pytest.mark.parametrize("dtype", DTYPES)
def test_a_mix_of_nan_and_inf_slices_resolves_each_on_its_own(dtype):
    array = np.array([[1.0, np.nan], [3.0, np.inf], [3.0, 4.0]], dtype=dtype)

    got = mt.Tensor(array, dtype=dtype).norm(2, 1).numpy()

    assert np.isnan(got[0])
    assert np.isinf(got[1])
    np.testing.assert_allclose(got[2], 5.0, rtol=1e-6)


@pytest.mark.parametrize("dtype", DTYPES)
def test_keepdim_keeps_the_infinity_in_place(dtype):
    array = np.array([[1.0, np.inf], [3.0, 4.0]], dtype=dtype)

    got = mt.Tensor(array, dtype=dtype).norm(2, 1, keepdim=True).numpy()

    assert got.shape == (2, 1)
    assert np.isinf(got[0, 0])
    np.testing.assert_allclose(got[1, 0], 5.0, rtol=1e-6)


# --- the guard that the scaling exists for, pinned alongside -----------------


def test_the_overflow_guard_still_holds_for_float32():
    """1e20 squared overflows f32, so a direct `sqrt(sum(x^2))` would report
    `inf` for a norm that fits comfortably. This is why the scaling is there,
    and it has to survive the fix above."""
    array = np.array([1e20, 1e20], dtype=np.float32)

    got = mt.Tensor(array).norm(2).numpy()

    assert np.isfinite(got), "the finite-scale overflow guard regressed"
    np.testing.assert_allclose(got, np.sqrt(2.0) * 1e20, rtol=1e-6)


def test_the_underflow_side_also_survives():
    """1e-25 squared is zero in f32, so the direct form would report 0."""
    array = np.array([1e-25, 1e-25], dtype=np.float32)

    got = mt.Tensor(array).norm(2).numpy()

    assert got > 0.0, "a representable tiny norm came back as zero"
    np.testing.assert_allclose(got, np.sqrt(2.0) * 1e-25, rtol=1e-6)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("p", SCALED_ORDERS)
def test_an_all_zero_slice_is_still_zero(p, dtype):
    """The other substituted scale. Dividing by the raw maximum would give
    `0 / 0`."""
    array = np.zeros(4, dtype=dtype)
    assert mt.Tensor(array, dtype=dtype).norm(p).numpy() == 0.0


@pytest.mark.parametrize("dtype", DTYPES)
def test_ordinary_values_are_untouched(dtype):
    """The substitution must not fire on anything finite and non-zero."""
    rng = np.random.default_rng(0)
    array = (rng.standard_normal((6, 7)) * 100).astype(dtype)
    tensor = mt.Tensor(array, dtype=dtype)
    reference = array.astype(np.float64)

    for p in SCALED_ORDERS:
        np.testing.assert_allclose(
            tensor.norm(p).numpy(), np.linalg.norm(reference.ravel(), ord=p), rtol=1e-5
        )
        for dim in (0, 1):
            np.testing.assert_allclose(
                tensor.norm(p, dim).numpy(),
                np.linalg.norm(reference, ord=p, axis=dim),
                rtol=1e-5,
            )


def test_norm_over_several_dims_at_once_sees_the_infinity():
    array = np.array([[[1.0, 2.0], [3.0, np.inf]]], dtype=np.float32)

    got = mt.Tensor(array).norm(2, [1, 2]).numpy()

    assert np.isinf(got).all()
