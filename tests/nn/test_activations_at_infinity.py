# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Activations answer their limits at the infinities, and so do their slopes.

`gelu`, `silu` and `mish` are `x` times a factor that vanishes as `x -> -inf`,
so at `-inf` the product was `-inf * 0 = NaN`; their gradients formed the same
product at both ends. `softsign` is `x / (1 + |x|)`, which is `inf / inf`.
`prelu` and `rrelu` took their negative part as `x - relu(x)`, which is
`inf - inf` at `+inf`. `relu`, `softplus`, `sigmoid` and `logsigmoid` already
gave their limits; these now do too. And `logsumexp` of a row that is `-inf`
throughout had the right value but a NaN gradient, where its derivative,
`softmax`, answers zero for that row.
"""

import math

import numpy as np
import pytest

import minitensor as mt
import minitensor.nn as nn

INF = math.inf
DTYPES = ["float32", "float64"]


def _run(fn, dtype):
    x = mt.tensor([-INF, INF, -50.0, 50.0, math.nan], dtype=dtype, requires_grad=True)
    y = fn(x)
    y.sum().backward()
    return y.numpy(), x.grad.numpy()


SMOOTH = [
    pytest.param(lambda x: mt.gelu(x), id="gelu"),
    pytest.param(lambda x: mt.gelu(x, approximate="tanh"), id="gelu-tanh"),
    pytest.param(lambda x: mt.silu(x), id="silu"),
    pytest.param(lambda x: mt.mish(x), id="mish"),
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fn", SMOOTH)
def test_smooth_rectifiers_reach_zero_and_identity(fn, dtype):
    values, grads = _run(fn, dtype)
    assert values[0] == 0.0 and values[1] == INF
    assert grads[0] == 0.0 and grads[1] == 1.0
    assert np.isnan(values[4])


@pytest.mark.parametrize("dtype", DTYPES)
def test_softsign_reaches_its_asymptotes(dtype):
    values, grads = _run(mt.softsign, dtype)
    assert values[:2].tolist() == [-1.0, 1.0]
    assert grads[:2].tolist() == [0.0, 0.0]
    assert np.isnan(values[4])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "fn",
    [
        pytest.param(
            lambda x: nn.prelu(x, mt.tensor([0.25], dtype=x.dtype)), id="prelu"
        ),
        pytest.param(lambda x: nn.rrelu(x, training=False), id="rrelu"),
    ],
)
def test_learned_and_random_leaks_keep_plus_infinity(fn, dtype):
    values, grads = _run(fn, dtype)
    assert values[0] == -INF and values[1] == INF
    assert grads[1] == 1.0


def test_prelu_still_takes_the_slope_at_zero():
    x = mt.tensor([0.0], requires_grad=True)
    nn.prelu(x, mt.tensor([0.25])).sum().backward()
    assert x.grad.tolist() == [0.25]


@pytest.mark.parametrize("dtype", DTYPES)
def test_logsumexp_of_a_masked_row_has_a_zero_gradient(dtype):
    x = mt.tensor(
        [[-INF, -INF], [0.0, -INF], [1.0, 2.0]], dtype=dtype, requires_grad=True
    )
    y = mt.logsumexp(x, 1)
    y.sum().backward()
    assert y.tolist()[:2] == [-INF, 0.0]
    grad = x.grad.numpy()
    assert grad[0].tolist() == [0.0, 0.0] and grad[1].tolist() == [1.0, 0.0]
    expected = np.exp([1.0, 2.0]) / np.exp([1.0, 2.0]).sum()
    assert np.allclose(grad[2], expected, rtol=1e-6)


def test_logsumexp_still_passes_gradcheck():
    rng = np.random.default_rng(5)
    x = mt.tensor(rng.standard_normal((3, 4)), dtype="float64", requires_grad=True)
    assert mt.gradcheck(lambda t: mt.logsumexp(t, 1).sum(), [x])


@pytest.mark.parametrize("x", [1e-1, 1e-2, 1e-4, 1e-8, -3e-3, 1e-20])
def test_tanhshrink_keeps_its_digits_near_zero(x):
    # `x - tanh(x)` is `x**3 / 3` there, and the subtraction cancelled it
    # away: in float32 the answer at 1e-4 came out negative and 22 times too
    # large. The leading terms of the series are the reference.
    expected = sum(
        c * x**p
        for c, p in (
            (1 / 3, 3),
            (-2 / 15, 5),
            (17 / 315, 7),
            (-62 / 2835, 9),
            (1382 / 155925, 11),
            (-21844 / 6081075, 13),
            (929569 / 638512875, 15),
        )
    )
    for dtype, tolerance in (("float32", 1e-6), ("float64", 1e-14)):
        got = mt.tanhshrink(mt.tensor([x], dtype=dtype)).numpy()[0]
        assert got == pytest.approx(expected, rel=tolerance), dtype


ELU_FORMS = [
    pytest.param(lambda t: mt.elu(t), lambda x: math.expm1(x), id="elu"),
    pytest.param(lambda t: nn.ELU()(t), lambda x: math.expm1(x), id="ELU-layer"),
    pytest.param(
        lambda t: mt.selu(t),
        lambda x: 1.0507009873554805 * 1.6732632423543772 * math.expm1(x),
        id="selu",
    ),
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fn,reference", ELU_FORMS)
@pytest.mark.parametrize("x", [-1e-3, -1e-8, -1e-12, -3.0])
def test_elu_family_keeps_its_digits_just_below_zero(fn, reference, x, dtype):
    # `exp(x) - 1` cancels every digit there: `elu(-1e-8)` in float32 was 0,
    # and the layer composed that same subtraction itself.
    got = fn(mt.tensor([x], dtype=dtype)).numpy()[0]
    assert got == pytest.approx(reference(x), rel=1e-6 if dtype == "float32" else 1e-14)
