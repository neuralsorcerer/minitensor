# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`clamp` with a bound per element, and `clip` on the same path as `clamp`.

A tensor bound used to be refused with "min must be a real number or None".
And `clip`, documented as `clamp` under another name, sent its bounds through
an f64, so on an int64 tensor it rounded a bound past 2**53 that `clamp`
honoured exactly.
"""

import numpy as np
import pytest

import minitensor as mt

X = np.array([[-2.0, 0.5, 3.0], [1.0, 4.0, -1.0]])
LO = np.array([0.0, 0.0, 1.0])
HI = np.array([1.0, 2.0, 2.0])


def _t(a, **kw):
    return mt.tensor(a, dtype="float64", **kw)


@pytest.mark.parametrize(
    "low,high",
    [(LO, HI), (LO, None), (None, HI), (0.25, HI), (LO, 1.5)],
)
def test_tensor_bounds_clamp_each_element(low, high):
    lo = _t(low) if isinstance(low, np.ndarray) else low
    hi = _t(high) if isinstance(high, np.ndarray) else high
    expected = X
    if low is not None:
        expected = np.maximum(expected, low)
    if high is not None:
        expected = np.minimum(expected, high)
    for got in (_t(X).clamp(lo, hi), mt.clamp(_t(X), lo, hi), _t(X).clip(lo, hi)):
        assert np.array_equal(got.numpy(), expected)


def test_the_bound_that_holds_an_element_receives_its_gradient():
    x = _t(X, requires_grad=True)
    lo = _t(LO, requires_grad=True)
    hi = _t(HI, requires_grad=True)
    x.clamp(lo, hi).sum().backward()
    inside = (X >= LO) & (X <= HI)
    assert np.array_equal(x.grad.numpy(), inside.astype(float))
    assert np.array_equal(lo.grad.numpy(), (X < LO).sum(0).astype(float))
    assert np.array_equal(hi.grad.numpy(), (X > HI).sum(0).astype(float))


def test_tensor_bounds_pass_gradcheck():
    rng = np.random.default_rng(3)
    weights = _t(rng.standard_normal((2, 3)))
    assert mt.gradcheck(
        lambda a, lo, hi: (a.clamp(lo, hi) * weights).sum(),
        [
            _t(rng.standard_normal((2, 3)), requires_grad=True),
            _t([-0.5, -0.3, -0.2], requires_grad=True),
            _t([0.4, 0.2, 0.3], requires_grad=True),
        ],
    )


def test_crossed_tensor_bounds_let_the_upper_one_win():
    got = _t([0.0, 5.0]).clamp(_t([3.0, 3.0]), _t([1.0, 1.0]))
    assert got.tolist() == [1.0, 1.0]


def test_crossed_scalar_bounds_are_still_refused():
    with pytest.raises(ValueError, match="cannot be greater than maximum"):
        _t(X).clamp(3.0, 1.0)


def test_an_integer_tensor_between_integer_tensor_bounds_stays_integer():
    x = mt.tensor([1, 5, 9], dtype="int32")
    got = x.clamp(
        mt.tensor([2, 2, 2], dtype="int32"), mt.tensor([4, 4, 4], dtype="int32")
    )
    assert got.dtype == "int32"
    assert got.tolist() == [2, 4, 4]


def test_a_bound_of_another_kind_is_a_type_error_naming_both_forms():
    with pytest.raises(TypeError, match="real number, a Tensor or None"):
        _t(X).clamp("low")


def test_clip_honours_an_int64_bound_past_two_to_the_fifty_three():
    x = mt.tensor([2**60, 2**60 + 3], dtype="int64")
    expected = [2**60 + 1, 2**60 + 3]
    assert x.clip(min=2**60 + 1).tolist() == expected
    assert mt.clip(x, 2**60 + 1, None).tolist() == expected
    assert x.clamp(min=2**60 + 1).tolist() == expected
