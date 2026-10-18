# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A tensor compared with one number of another dtype, exactly and in place.

`x > 0` on a float tensor is an integer comparison -- the literal is an exact
integer -- and the exact comparison of an integer with a float widened both
whole operands to int64 and float64 and ordered every pair through a floor: it
cost forty times what `x > 0.0` did. Against one number the question each
element is asked can be posed once, at the tensor's own dtype: an integer the
float dtype holds is itself, and one it does not lies strictly between two
neighbouring floats, so `x < n` is `x < above` and `x > n` is `x > below`,
and `x == n` is false throughout. An int32 tensor against any Python int
widened to int64 first, and is now posed the same way. The answers are the
exact ones still; these check them against Python's own exact arithmetic.
"""

from __future__ import annotations

import math
import operator
from fractions import Fraction

import numpy as np
import pytest

import minitensor as mt

OPERATORS = [
    operator.eq,
    operator.ne,
    operator.lt,
    operator.le,
    operator.gt,
    operator.ge,
]
SWAPPED = {
    operator.lt: operator.gt,
    operator.le: operator.ge,
    operator.gt: operator.lt,
    operator.ge: operator.le,
}


def _exact(op, a, b):
    if any(isinstance(v, float) and math.isnan(v) for v in (a, b)):
        return op is operator.ne
    if any(isinstance(v, float) and math.isinf(v) for v in (a, b)):
        return op(a, b)
    return op(Fraction(a), Fraction(b))


def _values(dtype):
    if dtype.startswith("float"):
        raw = [0.0, -0.0, 0.5, -1.0, 2.5, 16777216.0, 16777218.0, 2.0**53, 2.0**53 + 2]
        raw += [2.0**63, -(2.0**63), 1e-45, -1e-45, 3e38, math.inf, -math.inf, math.nan]
        return [float(np.array(v, dtype=dtype)) for v in raw]
    raw = [0, 1, -1, 3, 16777217, -16777217, 2**31 - 1, -(2**31)]
    if dtype == "int64":
        raw += [2**53 + 1, -(2**53) - 1, 2**63 - 1, -(2**63)]
    return raw


NUMBERS = {
    "int64": [0, 3, -1, 16777217, -16777217, 2**31, -(2**31) - 1, 2**53 + 1, 2**63 - 1],
    "int32": [0, 3, -1, 16777217, 2**31 - 1, -(2**31)],
    "float64": [0.5, -0.0, 2.0**53 + 2, 2.0**31, -(2.0**31) - 1.0, 2147483647.5, 0.1]
    + [1e300, 5e-324, 2.0**63, math.inf, -math.inf, math.nan],
    "float32": [0.5, -0.0, 16777218.0, 3e38, 1e-45, math.inf, math.nan],
}

PAIRS = [
    ("float32", "int64"),
    ("float32", "int32"),
    ("float32", "float64"),
    ("float64", "int64"),
    ("float64", "int32"),
    ("int32", "int64"),
    ("int32", "float64"),
    ("int32", "float32"),
    ("int64", "float64"),
    ("int64", "float32"),
    ("int64", "int32"),
]


@pytest.mark.parametrize("tensor_dtype,number_dtype", PAIRS)
@pytest.mark.parametrize("shape", [(), (1,), (1, 1)], ids=str)
def test_every_relation_is_the_exact_one(tensor_dtype, number_dtype, shape):
    values = _values(tensor_dtype)
    tensor = mt.Tensor(np.array(values, dtype=tensor_dtype), dtype=tensor_dtype)
    for number in NUMBERS[number_dtype]:
        single = mt.Tensor(
            np.full(shape, number, dtype=number_dtype), dtype=number_dtype
        )
        number = np.array(number, dtype=number_dtype).item()
        for op in OPERATORS:
            for got, want in [
                (op(tensor, single), [_exact(op, v, number) for v in values]),
                (
                    op(single, tensor),
                    [_exact(SWAPPED.get(op, op), v, number) for v in values],
                ),
            ]:
                got = got.numpy()
                assert got.shape == np.broadcast_shapes(shape, (len(values),))
                assert got.ravel().tolist() == want, (op.__name__, number)


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
def test_python_numbers_are_compared_exactly(dtype):
    values = _values(dtype)
    tensor = mt.Tensor(np.array(values, dtype=dtype), dtype=dtype)
    for number in [0, 3, 16777217, -(2**31) - 1, 2**53 + 1, 2.5, -0.5, 1e300]:
        if isinstance(number, float) and dtype.startswith("float"):
            # A Python float beside a float tensor is read at its precision.
            with np.errstate(over="ignore"):
                number = np.array(number, dtype=dtype).item()
        for op in OPERATORS:
            want = [_exact(op, v, number) for v in values]
            assert op(tensor, number).tolist() == want, (op.__name__, number)
            reflected = [_exact(SWAPPED.get(op, op), v, number) for v in values]
            assert op(number, tensor).tolist() == reflected, (op.__name__, number)


def test_tensors_of_mixed_dtypes_are_compared_exactly_too():
    # The pairs are widened as they are read now, rather than copied first.
    ints = [0, 1, -1, 2**53, 2**53 + 1, -(2**53) - 1, 2**63 - 1, -(2**63), 16777217]
    floats = [0.0, -0.0, 0.5, 2.0**53, 2.0**53 + 2, 2.0**63, 16777216.0, math.nan]
    floats += [math.inf, -math.inf]
    for int_dtype in ("int32", "int64"):
        iv = [v for v in ints if int_dtype == "int64" or -(2**31) <= v < 2**31]
        column = mt.Tensor(np.array(iv, dtype=int_dtype), dtype=int_dtype).reshape(
            [-1, 1]
        )
        for float_dtype in ("float32", "float64"):
            fv = [float(np.array(v, dtype=float_dtype)) for v in floats]
            row = mt.Tensor(np.array(fv, dtype=float_dtype), dtype=float_dtype)
            row = row.reshape([1, -1])
            for op in OPERATORS:
                want = [[_exact(op, a, b) for b in fv] for a in iv]
                assert op(column, row).tolist() == want, op.__name__
                swapped = SWAPPED.get(op, op)
                want = [[_exact(swapped, a, b) for b in fv] for a in iv]
                assert op(row, column).tolist() == want, op.__name__


WIDE = [2**63, -(2**63) - 1, 2**64 + 1, 10**30, -(10**30), 2**128 - 2**104]
WIDE += [2**128, 10**308 + 1, 10**400, -(10**400)]


def _tensor_values(dtype):
    if dtype == "bool":
        return [True, False]
    if dtype.startswith("int"):
        return _values(dtype)
    raw = [0.0, -1.0, 2.0**63, -(2.0**63), 2.0**64, 1e30, -1e30, 3.4e38, 1e308]
    raw += [math.inf, -math.inf, math.nan]
    with np.errstate(over="ignore"):
        return [float(np.array(v, dtype=dtype)) for v in raw]


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64", "bool"])
def test_an_int_past_int64_is_compared_exactly_rather_than_refused(dtype):
    # Beside an integer tensor it was refused -- `x < 2**63` raised -- and
    # beside a float one it was rounded to the tensor's width first, so the
    # float nearest it compared equal to it.
    values = _tensor_values(dtype)
    tensor = mt.Tensor(np.array(values, dtype=dtype), dtype=dtype)
    for number in WIDE:
        for op in OPERATORS:
            want = [_exact(op, v, number) for v in values]
            reflected = [_exact(SWAPPED.get(op, op), v, number) for v in values]
            function = getattr(mt, op.__name__)
            assert op(tensor, number).tolist() == want, (op.__name__, number)
            assert op(number, tensor).tolist() == reflected, (op.__name__, number)
            assert function(tensor, number).tolist() == want, (op.__name__, number)
            assert function(number, tensor).tolist() == reflected, op.__name__


def test_the_nearest_float_is_not_the_int():
    nearest = float(2**64 + 1)  # 2**64
    tensor = mt.Tensor([nearest], dtype="float64")
    assert (tensor == 2**64 + 1).tolist() == [False]
    assert (tensor < 2**64 + 1).tolist() == [True]
    assert (tensor == 2**64).tolist() == [True]
    assert (mt.Tensor([2**62]) < 2**63).tolist() == [True]
