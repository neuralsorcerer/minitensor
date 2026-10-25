# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""An integer beside a float is carried exactly and rounded once.

Promoted the ordinary way, the integer was rounded to the result's float dtype
before the operation rounded again: `16777217 + 16777218.0` in float32 came out
`33554432`, a third of int32 quotients were an ulp off, and an int64 timestamp
past 2^53 lost its last digits to `% 1e9`. Every case here is checked against
the exact rational result, rounded to nearest with ties to even.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

import minitensor as mt


def _float32(value: Fraction) -> float:
    """`value` rounded to the nearest float32, ties to even."""

    if value == 0:
        return 0.0
    magnitude = abs(value)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if Fraction(2) ** exponent > magnitude:
        exponent -= 1
    last = Fraction(2) ** max(exponent - 23, -149)
    units = magnitude / last
    whole = units.numerator // units.denominator
    rest = units - whole
    if rest > Fraction(1, 2) or (rest == Fraction(1, 2) and whole % 2 == 1):
        whole += 1
    rounded = whole * last
    result = math.inf if rounded >= 2**128 - 2**103 else float(rounded)
    return -result if value < 0 else result


def _float64(value: Fraction) -> float:
    return float(value)  # correctly rounded


_OPERATIONS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
}


def _floor(a, b):
    return Fraction(math.floor(a / b))


def _floored(a, b):
    return a - math.floor(a / b) * b


def _truncated(a, b):
    quotient = a / b
    whole = math.floor(quotient) if quotient >= 0 else math.ceil(quotient)
    return a - whole * b


_MODULI = {
    "//": (lambda a, b: a // b, _floor),
    "%": (lambda a, b: a % b, _floored),
    "fmod": (lambda a, b: a.fmod(b), _truncated),
}


def _assert_rounded_once(lhs, rhs, lhs_dtype, rhs_dtype, rounding, operations):
    left = mt.tensor(list(lhs), dtype=lhs_dtype)
    right = mt.tensor(list(rhs), dtype=rhs_dtype)
    for name, (apply, exact) in operations.items():
        got = apply(left, right).tolist()
        for a, b, value in zip(lhs, rhs, got):
            if b == 0:
                continue
            want = rounding(exact(Fraction(a), Fraction(b)))
            assert value == want, f"{a!r} {name} {b!r}: {value!r}, want {want!r}"


def _arithmetic(*names):
    return {name: (_OPERATIONS[name], _OPERATIONS[name]) for name in names}


def _ints(rng, low, high, n):
    return [int(v) for v in rng.integers(low, high, n, dtype=np.int64)]


def _float32s(rng, n, low=-30, high=40):
    signs = rng.choice([-1.0, 1.0], n)
    values = signs * rng.uniform(1, 2, n) * np.exp2(rng.integers(low, high, n))
    return [float(v) for v in values.astype(np.float32)]


def test_int32_beside_float32():
    rng = np.random.default_rng(401)
    ints = _ints(rng, -(2**31), 2**31, 2000)
    floats = _float32s(rng, 2000)
    every = _arithmetic("+", "-", "*", "/")
    _assert_rounded_once(ints, floats, "int32", "float32", _float32, every)
    _assert_rounded_once(floats, ints, "float32", "int32", _float32, every)


def test_int32_quotients():
    rng = np.random.default_rng(409)
    ints = _ints(rng, -(2**31), 2**31, 2000)
    divisors = [v or 1 for v in _ints(rng, -(2**31), 2**31, 2000)]
    _assert_rounded_once(ints, divisors, "int32", "int32", _float32, _arithmetic("/"))
    assert (mt.tensor([16777219], dtype="int32") / 1).tolist() == [16777220.0]


def test_halfway_integers_beside_tiny_floats():
    # An odd integer past 2^24 is halfway between two float32s; the tiny
    # float decides which way it goes, and rounded first it never could.
    rng = np.random.default_rng(419)
    halfway = [2 * int(v) + 1 for v in rng.integers(2**23, 2**24, 1000)]
    tiny = _float32s(rng, 1000, low=-60, high=-20)
    _assert_rounded_once(
        halfway, tiny, "int32", "float32", _float32, _arithmetic("+", "-")
    )
    sums = mt.tensor([16777217], dtype="int32") + mt.tensor([2.0**-40])
    assert sums.tolist() == [16777218.0]


def test_int64_past_two_to_the_53():
    rng = np.random.default_rng(421)
    ints = _ints(rng, -(2**63), 2**63, 2000)
    floats = [
        float(v)
        for v in rng.uniform(-1, 1, 2000) * np.exp2(rng.integers(-30, 70, 2000))
    ]
    every = _arithmetic("+", "-", "*", "/")
    _assert_rounded_once(ints, floats, "int64", "float64", _float64, every)
    _assert_rounded_once(floats, ints, "float64", "int64", _float64, every)
    divisors = [v or 3 for v in _ints(rng, -(2**40), 2**40, 2000)]
    _assert_rounded_once(ints, divisors, "int64", "int64", _float64, _arithmetic("/"))
    _assert_rounded_once(
        ints, _float32s(rng, 2000), "int64", "float32", _float32, every
    )
    # 2^53 + 1 is three times a whole number; rounded to 2^53 first, the
    # quotient came out half below it.
    third = mt.tensor([2**53 + 1], dtype="int64") / 3
    assert third.tolist() == [3002399751580331.0]


def test_floor_division_and_remainders():
    rng = np.random.default_rng(431)
    ints = _ints(rng, -(2**31), 2**31, 1500)
    floats = _float32s(rng, 1500)
    _assert_rounded_once(ints, floats, "int32", "float32", _float32, _MODULI)
    _assert_rounded_once(
        floats, [v or 1 for v in ints], "float32", "int32", _float32, _MODULI
    )
    # Quotients far past 2^53, and remainders far below the integer.
    tiny = _float32s(rng, 1500, low=-140, high=-100)
    _assert_rounded_once(ints, tiny, "int32", "float32", _float32, _MODULI)
    wide = _ints(rng, -(2**63), 2**63, 1500)
    doubles = [
        float(v)
        for v in rng.uniform(-1, 1, 1500) * np.exp2(rng.integers(-30, 70, 1500))
    ]
    _assert_rounded_once(wide, doubles, "int64", "float64", _float64, _MODULI)
    _assert_rounded_once(wide, floats, "int64", "float32", _float32, _MODULI)


def test_nanosecond_timestamps_keep_their_digits():
    stamps = [1_718_393_483_076_441_323, 1_650_621_728_992_911_547]
    tensor = mt.tensor(stamps, dtype="int64")
    assert (tensor % 1e9).tolist() == [76441323.0, 992911547.0]
    assert (tensor // 1e9).tolist() == [1718393483.0, 1650621728.0]
    assert (tensor / 1e9).tolist() == [float(Fraction(s, 10**9)) for s in stamps]


def test_python_numbers_beside_integer_tensors():
    rng = np.random.default_rng(433)
    ints = _ints(rng, -(2**31), 2**31, 2000)
    tensor = mt.tensor(ints, dtype="int32")
    assert (tensor + 1.5).tolist() == [_float32(a + Fraction(3, 2)) for a in ints]
    assert (tensor / 3).tolist() == [_float32(Fraction(a, 3)) for a in ints]
    assert (3.0 / tensor).tolist() == [_float32(Fraction(3, a)) for a in ints]
    tenth = Fraction(float(np.float32(0.1)))
    assert (tensor % 0.1).tolist() == [
        _float32(_floored(Fraction(a), tenth)) for a in ints
    ]


def test_specials_and_signed_zeros_follow_the_float_rules():
    ints = mt.tensor([4, -4, 0, 3], dtype="int32")
    zero_signs = [
        math.copysign(1.0, v)
        for v in (ints % mt.tensor([2.0, 2.0, -2.0, 1.0])).tolist()
    ]
    assert zero_signs == [1.0, 1.0, -1.0, 1.0]
    fmod_signs = [
        math.copysign(1.0, v)
        for v in ints.fmod(mt.tensor([-2.0, -2.0, -2.0, 1.0])).tolist()
    ]
    assert fmod_signs == [1.0, -1.0, 1.0, 1.0]
    assert (ints // mt.tensor([0.0, 0.0, 1.0, -0.0])).tolist() == [
        math.inf,
        -math.inf,
        0.0,
        -math.inf,
    ]
    assert all(math.isnan(v) for v in (ints % 0.0).tolist())
    infinite = mt.tensor([math.inf, -math.inf, math.inf, -math.inf])
    assert (ints // infinite).tolist() == [0.0, 0.0, 0.0, -1.0]
    assert (ints % infinite).tolist() == [4.0, -4.0, 0.0, -math.inf]
    assert (mt.tensor([1.0, -1.0]) // mt.tensor([0, 0], dtype="int32")).tolist() == [
        math.inf,
        -math.inf,
    ]
    products = (mt.tensor([0, 0], dtype="int32") * mt.tensor([-3.0, math.nan])).tolist()
    assert math.copysign(1.0, products[0]) == -1.0 and math.isnan(products[1])
    assert (
        mt.tensor([3, 0], dtype="int32") / mt.tensor([0, 0], dtype="int32")
    ).tolist()[0] == math.inf


def test_broadcast_and_strided_integer_operands():
    column = mt.tensor([[1], [2], [16777217]], dtype="int32")
    row = mt.tensor([[0.5, 1.0, -0.25, 3.0]])
    assert (column + row).tolist()[2] == [
        16777218.0,
        16777218.0,
        16777216.0,
        16777220.0,
    ]
    expanded = mt.tensor([16777217, 5], dtype="int32").reshape([2, 1]).expand([2, 3])
    assert (expanded + mt.tensor([0.0, 1.0, 2.0])).tolist() == [
        [16777216.0, 16777218.0, 16777220.0],
        [5.0, 6.0, 7.0],
    ]
    transposed = mt.tensor([[1, 2], [3, 16777219]], dtype="int32").t()
    assert (transposed / 1.0).tolist() == [[1.0, 3.0], [2.0, 16777220.0]]


def test_gradients_reach_the_float_operand():
    weights = mt.tensor([1.5, -2.0, 3.25], requires_grad=True)
    counts = mt.tensor([7, 3, -5], dtype="int32")
    (weights * counts + counts / weights + weights % counts).sum().backward()
    assert str(weights.grad.dtype) == "float32"
    want = [k - k / (w * w) + 1 for w, k in zip([1.5, -2.0, 3.25], [7, 3, -5])]
    assert np.allclose(weights.grad.tolist(), want, rtol=1e-6)


def test_in_place_float_beside_integer():
    values = mt.tensor([1.0, 2.0])
    values += mt.tensor([16777217, 1], dtype="int32")
    assert values.tolist() == [16777218.0, 3.0]


def test_python_ints_float32_cannot_hold():
    x = mt.tensor([1.0, 0.5])
    assert (x + 16777217).tolist() == [16777218.0, 16777218.0]
    assert (16777217 + x).tolist() == [16777218.0, 16777218.0]
    assert (x - 16777217).tolist() == [-16777216.0, -16777216.0]
    assert (16777217 / mt.tensor([1.0, 3.0])).tolist() == [16777216.0, 5592405.5]
    assert (16777219 % mt.tensor([2.0, 4.0])).tolist() == [1.0, 3.0]
    assert (x + [16777217, 16777219]).tolist() == [16777218.0, 16777220.0]
    assert str((x + 16777217).dtype) == "float32"
    wide = mt.tensor([1.0], dtype="float64")
    assert (wide + (2**53 + 1)).tolist() == [9007199254740994.0]
    # An int the dtype holds is read at its width, as before.
    assert (x * 2).tolist() == [2.0, 1.0]
    assert (x + [1, 2]).tolist() == [2.0, 2.5]


def test_differences_of_integers_beside_float_ends():
    big = mt.tensor([16777217, 16777219, 16777222], dtype="int32")
    prepended = mt.diff(big, prepend=mt.tensor([0.5]))
    assert prepended.tolist() == [16777216.0, 2.0, 3.0]
    assert str(prepended.dtype) == "float32"
    assert mt.diff(big, n=2, append=mt.tensor([16777226.0])).tolist() == [1.0, 1.0]
    assert mt.ediff1d(big, to_end=mt.tensor([0.5])).tolist() == [2.0, 3.0, 0.5]
    # An int32 difference past the int32 range is taken exactly.
    extremes = mt.tensor([2**31 - 1, -(2**31)], dtype="int32")
    assert mt.ediff1d(extremes, to_begin=0.5).tolist() == [0.5, -4294967296.0]
    assert str(mt.ediff1d(big, to_end=7).dtype) == "int32"
