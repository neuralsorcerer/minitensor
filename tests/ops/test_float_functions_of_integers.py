# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A real-valued function of an int32 sees the exact integer.

An int32 input widened to float32 was rounded before the function saw it, and
for a periodic function that is not an ulp but the wrong answer:
`sin(16777217)` is 0.106, and came out `sin(16777216)`, -0.779. Evaluated in
float64, which holds every int32, the result is rounded to float32 once. The
same holds for the binary functions whose result is real, for a power with an
integer beside a float, and for a Python int the float dtype cannot hold.
"""

from __future__ import annotations

import math

import numpy as np

import minitensor as mt


def _float32(value: float) -> float:
    return float(np.float32(value))


def _ints(seed, n=2000):
    rng = np.random.default_rng(seed)
    return [int(v) for v in rng.integers(-(2**31), 2**31, n)]


def test_periodic_functions_of_large_int32():
    tensor = mt.tensor([16777217], dtype="int32")
    assert tensor.sin().tolist() == [_float32(math.sin(16777217))]
    assert abs(tensor.sin().item() - 0.10583) < 1e-4
    ints = _ints(503)
    tensor = mt.tensor(ints, dtype="int32")
    for name, reference in [("sin", math.sin), ("cos", math.cos), ("tan", math.tan)]:
        got = getattr(tensor, name)().tolist()
        assert got == [_float32(reference(v)) for v in ints], name


def test_real_functions_of_int32_round_once():
    ints = [abs(v) + 1 for v in _ints(509)]
    tensor = mt.tensor(ints, dtype="int32")
    cases = [
        ("sqrt", math.sqrt),
        ("log", math.log),
        ("log1p", math.log1p),
        ("atan", math.atan),
        ("tanh", math.tanh),
    ]
    for name, reference in cases:
        got = getattr(tensor, name)().tolist()
        assert got == [_float32(reference(v)) for v in ints], name
    assert str(tensor.sqrt().dtype) == "float32"
    assert str(mt.tensor([4], dtype="int64").sqrt().dtype) == "float64"


def test_reductions_of_int32_see_the_exact_values():
    tensor = mt.tensor([16777217, 1], dtype="int32")
    assert tensor.norm().item() == _float32(math.hypot(16777217, 1))
    assert tensor.norm().item() == 16777218.0
    ints = [16777217, 16777219, 16777221]
    exact_mean = sum(ints) / len(ints)
    assert mt.tensor(ints, dtype="int32").mean().item() == _float32(exact_mean)


def test_binary_real_functions_of_integers():
    rng = np.random.default_rng(521)
    left = [int(v) for v in rng.integers(2**24, 2**31, 1000)]
    right = [int(v) for v in rng.integers(1, 2**10, 1000)]
    lhs = mt.tensor(left, dtype="int32")
    rhs = mt.tensor(right, dtype="int32")
    pairs = list(zip(left, right))
    assert mt.hypot(lhs, rhs).tolist() == [_float32(math.hypot(a, b)) for a, b in pairs]
    assert mt.atan2(rhs, lhs).tolist() == [_float32(math.atan2(b, a)) for a, b in pairs]
    floats = mt.tensor([float(v) for v in right])
    assert mt.hypot(floats, lhs).tolist() == [
        _float32(math.hypot(b, a)) for a, b in pairs
    ]
    assert lhs.hypot(rhs).tolist() == [_float32(math.hypot(a, b)) for a, b in pairs]
    assert mt.hypot(mt.tensor([16777217], dtype="int32"), 1).tolist() == [16777218.0]


def test_nextafter_steps_in_the_result_dtype():
    # A neighbour in float32's own grid: the integer is converted first, and
    # a step taken in float64 would round back to where it started.
    assert mt.nextafter(
        mt.tensor([3], dtype="int32"), mt.tensor([1], dtype="int32")
    ).tolist() == [float(np.nextafter(np.float32(3), np.float32(1)))]


def test_powers_of_and_by_integers():
    x = mt.tensor([-1.0, 1.0000001, 2.0])
    by_int = (x ** mt.tensor([16777217] * 3, dtype="int32")).tolist()
    assert by_int[0] == -1.0
    assert (x**16777217).tolist() == by_int
    assert (x**2).tolist() == [1.0, _float32(float(np.float32(1.0000001)) ** 2), 4.0]
    roots = (mt.tensor([16777217, 2], dtype="int32") ** 0.5).tolist()
    assert roots == [_float32(math.sqrt(16777217)), _float32(math.sqrt(2))]
    assert (16777217 ** mt.tensor([1.0, 0.5])).tolist() == [16777216.0, 4096.0]
    # Two integers keep an integer power.
    assert str((mt.tensor([3], dtype="int32") ** 2).dtype) == "int32"


def test_gradients_reach_a_float_beside_an_integer():
    w = mt.tensor([3.0, 4.0], requires_grad=True)
    mt.hypot(w, mt.tensor([4, 3], dtype="int32")).sum().backward()
    assert str(w.grad.dtype) == "float32"
    assert np.allclose(w.grad.tolist(), [0.6, 0.8])
    v = mt.tensor([2.0], requires_grad=True)
    (v ** mt.tensor([3], dtype="int32")).sum().backward()
    assert np.allclose(v.grad.tolist(), [12.0])
