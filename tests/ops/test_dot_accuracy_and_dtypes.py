# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A dot product is a sum, and it was not accumulating like one.

`dot` multiplied the two operands and fed the products to `Iterator::sum`,
which is a single dependent chain of multiply-adds. That is one rounding per
element, so the error grew with the length -- on 65,536 float32 values it
measured 1.7e-6 against NumPy's 4.8e-8, a factor of 36 -- while every other
reduction in the library had already been moved onto the blocked pairwise
accumulation in `ops::util`.

The same chain is also why it could not vectorize: floating point addition is
not associative, so the compiler may not split one accumulator into eight
without being told to. It ran 7.5 times slower than NumPy's `sdot` on the same
65,536 elements.

`simd_dot_f32`/`simd_dot_f64` supply the eight (four, for f64) lanes, and
`accurate_pair_sum` blocks and folds them. The tests below pin the accuracy
against NumPy, which is the implementation being matched, and pin the parts
that must not have moved: the wrapping integer accumulation, the dtype
promotion, and the gradient.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

# 8192 is the block size and 131072 the point where the reduction goes
# parallel; both bracket a boundary the answer must not jump across.
LENGTHS = [1, 7, 8, 1024, 8191, 8192, 8193, 65_536, 131_071, 131_072, 1_000_000]


def _pair(n, seed=5):
    """All-positive operands: nothing cancels, so the rounding errors pile up
    in one direction instead of partly undoing each other. A signed dot product
    hides this defect behind its own cancellation."""
    rng = np.random.default_rng(seed)
    a = (rng.random(n) + 0.5).astype(np.float32)
    b = (rng.random(n) + 0.5).astype(np.float32)
    return a, b


@pytest.mark.parametrize("n", LENGTHS)
def test_dot_is_at_least_as_accurate_as_numpy(n):
    a, b = _pair(n)
    exact = float(a.astype(np.float64) @ b.astype(np.float64))
    ours = mt.Tensor(a, dtype="float32").dot(mt.Tensor(b, dtype="float32")).item()

    ours_error = abs(ours - exact) / exact
    numpy_error = abs(float(a @ b) - exact) / exact
    # The floor is a couple of float32 ulps: below that the two differ only by
    # which valid ordering they chose, and neither is the more correct. The
    # defect this pins was 1.7e-6 on 65536 elements, ten times above it.
    assert ours_error <= max(
        numpy_error * 3.0, 2e-7
    ), f"ours {ours_error:.3e} numpy {numpy_error:.3e}"


@pytest.mark.parametrize("n", [65_536, 1_000_000])
def test_dot_does_not_degrade_with_length(n):
    """The property the blocking exists for. A running float32 total over the
    same products is the thing being improved on, so it is computed here rather
    than assumed."""
    a, b = _pair(n, seed=9)
    exact = float(a.astype(np.float64) @ b.astype(np.float64))
    ours = mt.Tensor(a, dtype="float32").dot(mt.Tensor(b, dtype="float32")).item()

    prefix = 20_000
    naive = np.float32(0.0)
    for x, y in zip(a[:prefix], b[:prefix]):
        naive = np.float32(naive + np.float32(x * y))
    prefix_exact = float(a[:prefix].astype(np.float64) @ b[:prefix].astype(np.float64))

    assert abs(ours - exact) / exact < abs(naive - prefix_exact) / prefix_exact


@pytest.mark.parametrize("n", [8192, 131_072])
def test_float64_dot_is_at_least_as_accurate_as_numpy(n):
    rng = np.random.default_rng(11)
    a = rng.random(n) + 0.5
    b = rng.random(n) + 0.5
    # float64 has no wider type to check against, so the reference is Kahan
    # summation of the same products.
    total = 0.0
    comp = 0.0
    for x, y in zip(a, b):
        term = x * y - comp
        run = total + term
        comp = (run - total) - term
        total = run

    ours = mt.Tensor(a, dtype="float64").dot(mt.Tensor(b, dtype="float64")).item()
    ours_error = abs(ours - total) / total
    numpy_error = abs(float(a @ b) - total) / total
    assert ours_error <= max(numpy_error * 3.0, 4e-16)


@pytest.mark.parametrize("n", [3, 8192, 131_072])
def test_dot_still_matches_the_elementwise_product_summed(n):
    """Two spellings of one computation. `(a * b).sum()` goes through the
    ordinary reduction; `a.dot(b)` has its own kernel."""
    a, b = _pair(n, seed=13)
    ta = mt.Tensor(a, dtype="float32")
    tb = mt.Tensor(b, dtype="float32")
    direct = ta.dot(tb).item()
    composed = (ta * tb).sum().item()
    assert abs(direct - composed) <= 1e-6 * abs(composed)


@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_integer_dot_is_exact(dtype):
    rng = np.random.default_rng(17)
    a = rng.integers(-1000, 1000, size=100_000).astype(dtype)
    b = rng.integers(-1000, 1000, size=100_000).astype(dtype)
    expected = int(a.astype(np.int64) @ b.astype(np.int64))
    got = mt.Tensor(a, dtype=dtype).dot(mt.Tensor(b, dtype=dtype)).item()
    assert got == expected


def _wrap32(v):
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v


def test_integer_dot_still_wraps_rather_than_panicking():
    """Integer accumulation wraps deliberately -- both the product and the
    running total. Routing it through the blocked sum must not have turned that
    into an overflow panic in a checked build, nor changed the value: wrapping
    addition is associative, so blocking cannot move the answer."""
    n = 4096
    value = 46341  # 46341 squared is just past 2**31, so every product wraps
    a = np.full(n, value, dtype=np.int32)
    got = mt.Tensor(a, dtype="int32").dot(mt.Tensor(a, dtype="int32")).item()

    term = _wrap32(value * value)
    expected = 0
    for _ in range(n):
        expected = _wrap32(expected + term)
    assert got == expected


def test_dot_promotes_mixed_dtypes_the_way_arithmetic_does():
    a = np.array([1, 2, 3], dtype=np.int32)
    b = np.array([0.5, 0.25, 0.125], dtype=np.float32)
    got = mt.Tensor(a, dtype="int32").dot(mt.Tensor(b, dtype="float32"))
    assert got.dtype == "float32"
    assert got.item() == pytest.approx(1.375)


def test_dot_rejects_shapes_that_are_not_one_dimensional():
    v = mt.Tensor(np.ones((2, 2), dtype=np.float32), dtype="float32")
    with pytest.raises(Exception):
        v.dot(v)
    a = mt.Tensor(np.ones(3, dtype=np.float32), dtype="float32")
    b = mt.Tensor(np.ones(4, dtype=np.float32), dtype="float32")
    with pytest.raises(Exception):
        a.dot(b)


def test_dot_of_empty_vectors_is_zero():
    empty = mt.Tensor(np.zeros(0, dtype=np.float32), dtype="float32")
    assert empty.dot(empty).item() == 0.0


def test_dot_is_reproducible_across_repeated_calls():
    """The partials are collected in index order and folded pairwise, so the
    answer may not shift with however rayon happened to schedule the work."""
    a, b = _pair(2_000_000, seed=19)
    ta = mt.Tensor(a, dtype="float32")
    tb = mt.Tensor(b, dtype="float32")
    first = ta.dot(tb).item()
    assert all(ta.dot(tb).item() == first for _ in range(20))


def test_dot_gradient_is_unchanged():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    ta = mt.Tensor(a, dtype="float32", requires_grad=True)
    tb = mt.Tensor(b, dtype="float32", requires_grad=True)
    out = ta.dot(tb)
    out.backward()
    np.testing.assert_allclose(ta.grad.numpy(), b)
    np.testing.assert_allclose(tb.grad.numpy(), a)


def test_dot_propagates_non_finite_values():
    a = np.array([1.0, np.inf, 3.0], dtype=np.float32)
    b = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    got = mt.Tensor(a, dtype="float32").dot(mt.Tensor(b, dtype="float32")).item()
    assert np.isnan(got)  # inf * 0

    a = np.array([1.0, np.nan], dtype=np.float32)
    b = np.array([1.0, 1.0], dtype=np.float32)
    assert np.isnan(
        mt.Tensor(a, dtype="float32").dot(mt.Tensor(b, dtype="float32")).item()
    )
