# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A float has to fit the integer dtype it is written into, and a clamp with a
fractional bound widens an integer tensor.

A Python int that does not fit an integer dtype was already refused. A Python
float bound for one is truncated, as any float-to-integer conversion is -- but
NaN, an infinity or a value whose truncation is out of range has no integer to
truncate to, and the cast saturated them: `fill_(inf)` on int32 wrote
2147483647 and `full(..., nan, dtype="int32")` wrote zeros, with nothing to say
so. Every way of writing a Python number into an integer tensor now refuses
them with the message an oversized int gets.

`clamp` on an integer tensor truncated a fractional bound onto the integer:
`clamp([1, 5], 0, 2.5)` came back `[1, 2]`. A clamp answers with a bound or the
value itself, so an integer has an integer answer only when the bounds are
whole -- the rule `hardtanh` already followed. Fractional bounds now widen, as
they do there.
"""

import math

import numpy as np
import pytest

import minitensor as mt

UNFIT = {
    "int32": [math.inf, -math.inf, math.nan, 2147483648.0, -2147483649.0, 1e20],
    "int64": [math.inf, math.nan, 9.3e18, -1e19],
}

WRITERS = {
    "fill_": lambda dtype, v: mt.zeros((2,), dtype=dtype).fill_(v),
    "full": lambda dtype, v: mt.full((2,), v, dtype=dtype),
    "full_like": lambda dtype, v: mt.full_like(mt.zeros((2,), dtype=dtype), v),
    "new_full": lambda dtype, v: mt.zeros((1,), dtype=dtype).new_full((2,), v),
    "tensor": lambda dtype, v: mt.tensor([1.0, v], dtype=dtype),
    "setitem": lambda dtype, v: mt.zeros((2,), dtype=dtype).__setitem__(0, v),
}


@pytest.mark.parametrize("writer", sorted(WRITERS))
@pytest.mark.parametrize(
    "dtype,value", [(d, v) for d, values in UNFIT.items() for v in values]
)
def test_a_float_with_no_integer_to_truncate_to_is_refused(writer, dtype, value):
    with pytest.raises(OverflowError, match=f"does not fit in {dtype}"):
        WRITERS[writer](dtype, value)


@pytest.mark.parametrize("writer", ["fill_", "full", "full_like", "new_full"])
def test_a_float_that_fits_still_truncates(writer):
    out = WRITERS[writer]("int32", -2.9)
    assert out.dtype == "int32"
    assert out.tolist() == [-2, -2]
    assert WRITERS[writer]("int32", 2147483647.9).tolist() == [2147483647] * 2


def test_new_full_writes_a_large_integer_exactly():
    # It took the value as a float, so int64 values past 2**53 were rounded.
    value = 2**60 + 1
    out = mt.zeros((1,), dtype="int64").new_full((2,), value)
    assert out.tolist() == [value, value]


def test_a_float_tensor_still_takes_nan_and_infinity():
    assert np.isnan(mt.full((1,), math.nan).numpy()).all()
    assert mt.zeros((1,)).fill_(math.inf).tolist() == [math.inf]


@pytest.mark.parametrize(
    "call,expected",
    [
        (lambda t: t.clamp(0, 2.5), [0.0, 1.0, 2.5]),
        (lambda t: t.clamp(min=0.5), [0.5, 1.0, 5.0]),
        (lambda t: t.clamp_max(2.5), [0.0, 1.0, 2.5]),
        (lambda t: t.clamp_min(0.5), [0.5, 1.0, 5.0]),
        (lambda t: mt.clamp(t, -0.5, 4), [0.0, 1.0, 4.0]),
        (lambda t: mt.clip(t, 0.5, 3), [0.5, 1.0, 3.0]),
    ],
)
@pytest.mark.parametrize("dtype,widened", [("int32", "float32"), ("int64", "float64")])
def test_a_fractional_clamp_bound_widens_an_integer(call, expected, dtype, widened):
    out = call(mt.tensor([0, 1, 5], dtype=dtype))
    assert out.dtype == widened
    assert out.tolist() == expected


@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_a_whole_or_infinite_clamp_bound_keeps_the_integer(dtype):
    t = mt.tensor([0, 1, 5], dtype=dtype)
    for out, expected in [
        (t.clamp(1, 3), [1, 1, 3]),
        (t.clamp(2.0, 4.0), [2, 2, 4]),
        (t.clamp(0, math.inf), [0, 1, 5]),
        (t.clamp(-math.inf, 2), [0, 1, 2]),
    ]:
        assert out.dtype == dtype
        assert out.tolist() == expected
