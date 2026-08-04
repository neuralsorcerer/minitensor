# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`nn.GELU` and the function it computes.

The layer computes the tanh approximation,
`0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3)))`, which is *not* what the
functional `mt.gelu` computes -- that one is the exact `0.5x(1 + erf(x/sqrt2))`.
They differ by about 5e-4, which is the point of the approximation, so the two
cannot be swapped for one another.

The layer used to build its approximation out of nine separate tensor
operations, three of them broadcasting a cached scalar: nine passes over the
input and nine full-size allocations for one elementwise function. It now calls
the vectorised kernel that already implements the same formula. These tests pin
the values, so the two forms cannot drift apart again, and pin the distinction
from the exact form, so the layer is not "fixed" by pointing it at `mt.gelu`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import minitensor as mt
from minitensor import nn


def _tanh_gelu(values: np.ndarray) -> np.ndarray:
    inner = math.sqrt(2.0 / math.pi) * (values + 0.044715 * values**3)
    return 0.5 * values * (1.0 + np.tanh(inner))


def _exact_gelu(values: np.ndarray) -> np.ndarray:
    return np.array(
        [0.5 * v * (1.0 + math.erf(v / math.sqrt(2.0))) for v in values.ravel()]
    ).reshape(values.shape)


@pytest.mark.parametrize("dtype,tol", [("float32", 2e-6), ("float64", 1e-12)])
def test_gelu_layer_computes_the_tanh_approximation(dtype, tol):
    rng = np.random.default_rng(11)
    values = (rng.standard_normal(4000) * 3.0).astype(dtype)

    got = nn.GELU()(mt.Tensor(values, dtype=dtype)).numpy()

    np.testing.assert_allclose(
        got, _tanh_gelu(values.astype(np.float64)), atol=tol, rtol=0
    )


def test_the_layer_and_the_functional_form_are_different_functions():
    # Not a bug to be fixed by making them agree: one is the approximation.
    values = np.linspace(-4.0, 4.0, 401).astype(np.float32)
    tensor = mt.Tensor(values)

    layer = nn.GELU()(tensor).numpy()
    functional = mt.gelu(tensor).numpy()

    assert np.abs(layer - functional).max() > 1e-4
    np.testing.assert_allclose(
        functional, _exact_gelu(values.astype(np.float64)), atol=2e-7
    )


@pytest.mark.parametrize("shape", [(7,), (4, 5), (2, 3, 4), (1, 1, 1, 9)])
def test_gelu_layer_handles_any_shape(shape):
    rng = np.random.default_rng(12)
    values = rng.standard_normal(shape).astype(np.float32)

    got = nn.GELU()(mt.Tensor(values)).numpy()

    assert got.shape == shape
    np.testing.assert_allclose(
        got, _tanh_gelu(values.astype(np.float64)), atol=2e-6, rtol=0
    )


def test_gelu_layer_gradient_matches_central_differences():
    rng = np.random.default_rng(13)
    values = rng.standard_normal((5, 7))

    tensor = mt.Tensor(values, dtype="float64").requires_grad_(True)
    out = nn.GELU()(tensor)
    upstream = rng.standard_normal((5, 7))
    mt.sum(out * mt.Tensor(upstream, dtype="float64")).backward()
    analytic = mt.get_gradient(tensor).numpy()

    eps = 1e-6
    for index in np.ndindex(*values.shape):
        plus, minus = values.copy(), values.copy()
        plus[index] += eps
        minus[index] -= eps
        numeric = ((_tanh_gelu(plus) - _tanh_gelu(minus)) * upstream).sum() / (2 * eps)
        assert abs(numeric - analytic[index]) < 1e-6 * max(1.0, abs(numeric)), index


def test_gelu_layer_saturates_without_overflowing():
    extreme = np.array([-1e4, -100.0, -20.0, 0.0, 20.0, 100.0, 1e4], dtype=np.float32)

    got = nn.GELU()(mt.Tensor(extreme)).numpy()

    assert np.isfinite(got).all()
    np.testing.assert_allclose(got[:3], 0.0, atol=1e-6)  # far negative -> 0
    np.testing.assert_allclose(got[4:], extreme[4:], rtol=1e-6)  # far positive -> x
