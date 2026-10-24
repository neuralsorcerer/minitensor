# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""A mean divides its exact sum by its exact count, and rounds once.

A float32 count past `2^24` is not always a float32, and dividing by its
rounding was an error on top of the sum's; an int32 mean converted its exact
int64 sum to float32 before dividing, a second rounding. The mean of `2^24 + 1`
threes came out 3.0000002. A float32 mean summed in float32, a rounding per
addition against a growing total, which put offset data tens of ulps out and
the mean of a constant off the constant. It is now summed in float64 and the
quotient rounded once: the float32 nearest the exact mean.
"""

from __future__ import annotations

import numpy as np

import minitensor as mt

LONG = 2**24 + 1


def test_an_int32_mean_rounds_once():
    assert mt.Tensor(np.full(LONG, 3, np.int32), dtype="int32").mean().item() == 3.0
    values = np.array([2**24 + 1, 2**24 + 2], dtype=np.int32)
    # The exact mean is 2^24 + 1.5, a tie between two float32s that goes to
    # the even one.
    got = mt.Tensor(values, dtype="int32").mean().item()
    assert got == float(np.float32(2**24 + 2))


def test_long_float32_means():
    ones = np.full(LONG, 1.5, np.float32)
    assert mt.Tensor(ones).mean().item() == 1.5
    assert mt.Tensor(ones, requires_grad=True).mean().item() == 1.5
    halves = np.full(LONG, 0.5, np.float32)
    halves[::2] = np.nan
    for tensor in (mt.Tensor(halves), mt.Tensor(halves, requires_grad=True)):
        assert tensor.nanmean().item() == 0.5


def test_long_float32_variances():
    constant = np.full(LONG, 0.25, np.float32)
    assert mt.Tensor(constant).var().item() == 0.0
    assert mt.Tensor(constant, requires_grad=True).var().item() == 0.0
    assert mt.Tensor(constant, requires_grad=True).nanvar().item() == 0.0
    assert mt.Tensor(constant).std().item() == 0.0
    one_hot = np.zeros(LONG, np.float32)
    one_hot[0] = 1.0
    want = float(np.float32((1 - 1 / LONG) / (LONG - 1)))
    for got in (
        mt.Tensor(one_hot).var().item(),
        mt.Tensor(one_hot, requires_grad=True).var().item(),
        mt.Tensor(one_hot).nanvar().item(),
        mt.Tensor(one_hot, requires_grad=True).nanvar().item(),
    ):
        assert abs(got - want) <= float(np.spacing(np.float32(want))), got


def test_the_nan_mean_with_a_gradient_sums_as_the_one_without():
    rng = np.random.default_rng(233)
    values = (1e4 + rng.normal(0, 1, 100_000)).astype(np.float32)
    values[::7] = np.nan
    plain = mt.Tensor(values).nanmean().item()
    traced = mt.Tensor(values, requires_grad=True).nanmean().item()
    exact = float(np.nanmean(values.astype(np.float64)))
    spacing = float(np.spacing(np.float32(exact)))
    assert abs(plain - exact) <= spacing and abs(traced - exact) <= spacing


def _exact_mean(values) -> float:
    from fractions import Fraction

    return float(sum(Fraction(float(v)) for v in values) / len(values))


def test_float32_means_are_correctly_rounded():
    rng = np.random.default_rng(239)
    for offset in (0.0, 1.0, 1e4):
        for n in (2, 3, 17, 1000, 2500):
            values = (offset + rng.normal(0, 1, n)).astype(np.float32)
            want = float(np.float32(_exact_mean(values)))
            assert mt.Tensor(values).mean().item() == want, (offset, n)
            assert mt.Tensor(values, requires_grad=True).mean().item() == want


def test_the_mean_of_a_constant_is_the_constant():
    for value in (0.1, 1.7632415, 3.3, -1e30):
        for n in (3, 7, 1000, 12345):
            values = np.full(n, value, np.float32)
            assert mt.Tensor(values).mean().item() == float(np.float32(value))
            assert mt.Tensor(values.reshape(1, -1)).mean(dim=1).item() == float(
                np.float32(value)
            )
            assert mt.Tensor(values.reshape(-1, 1)).mean(dim=0).item() == float(
                np.float32(value)
            )
            assert mt.Tensor(values).nanmean().item() == float(np.float32(value))


def test_every_layout_and_its_gradient():
    rng = np.random.default_rng(241)
    values = rng.normal(5, 1, (37, 53, 11)).astype(np.float32)
    wide = values.astype(np.float64)
    for dims in (None, [0], [1], [2], [0, 2], [1, 2], [0, 1]):
        for keepdim in (False, True):
            axes = None if dims is None else tuple(dims)
            want = wide.mean(axis=axes, keepdims=keepdim).astype(np.float32)
            plain = mt.Tensor(values).mean(dim=dims, keepdim=keepdim).numpy()
            tensor = mt.Tensor(values, requires_grad=True)
            traced = tensor.mean(dim=dims, keepdim=keepdim)
            np.testing.assert_array_equal(plain, traced.detach().numpy())
            np.testing.assert_allclose(plain, want, rtol=1.2e-7)
            traced.sum().backward()
            count = (
                values.size
                if dims is None
                else int(np.prod([values.shape[d] for d in dims]))
            )
            np.testing.assert_array_equal(
                tensor.grad.numpy(), np.full(values.shape, np.float32(1 / count))
            )
