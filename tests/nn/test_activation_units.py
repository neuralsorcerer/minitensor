# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""The named activation units: `relu6`, `hardtanh`, `hardsigmoid`, `hardswish`,
`mish`, `celu`, `logsigmoid`, `softshrink`, `tanhshrink`, `threshold`, `softmin`.

Each is checked against its own definition rather than against another
implementation of itself, and separately at the magnitudes where writing the
definition out directly overflows.
"""

import numpy as np
import pytest

import minitensor as mt


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _hardsigmoid(x):
    return np.clip(x / 6.0 + 0.5, 0.0, 1.0)


# name -> (call, reference). Anything with parameters is bound here so the rest
# of the file can treat them all alike.
UNITS = {
    "relu6": (lambda t: t.relu6(), lambda a: np.clip(a, 0.0, 6.0)),
    "hardtanh": (lambda t: t.hardtanh(-2.0, 3.0), lambda a: np.clip(a, -2.0, 3.0)),
    "hardsigmoid": (lambda t: t.hardsigmoid(), _hardsigmoid),
    "hardswish": (lambda t: t.hardswish(), lambda a: a * _hardsigmoid(a)),
    "mish": (lambda t: t.mish(), lambda a: a * np.tanh(np.log1p(np.exp(a)))),
    "celu": (
        lambda t: t.celu(1.5),
        lambda a: np.where(a > 0, a, 1.5 * np.expm1(a / 1.5)),
    ),
    "logsigmoid": (lambda t: t.logsigmoid(), lambda a: np.log(_sigmoid(a))),
    "softshrink": (
        lambda t: t.softshrink(0.5),
        lambda a: np.sign(a) * np.maximum(np.abs(a) - 0.5, 0.0),
    ),
    "tanhshrink": (lambda t: t.tanhshrink(), lambda a: a - np.tanh(a)),
    "threshold": (
        lambda t: t.threshold(0.5, -1.0),
        lambda a: np.where(a > 0.5, a, -1.0),
    ),
}

# The free-function argument lists, so `mt.<name>` and `mt.functional.<name>`
# can be checked against the method with the same parameters bound.
FREE_ARGS = {
    "hardtanh": (-2.0, 3.0),
    "celu": (1.5,),
    "softshrink": (0.5,),
    "threshold": (0.5, -1.0),
}

# Spans every branch: both flats of each hard unit, the sloped middles, the
# shrink bands, and the exact breakpoints.
SAMPLE = np.array(
    [-8.0, -6.0, -3.0, -2.0, -0.5, -0.25, 0.0, 0.25, 0.5, 2.0, 3.0, 6.0, 8.0]
)


@pytest.mark.parametrize("name", sorted(UNITS))
def test_matches_its_definition(name):
    call, reference = UNITS[name]
    got = call(mt.from_numpy(SAMPLE.copy())).numpy()
    np.testing.assert_allclose(got, reference(SAMPLE), rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("name", sorted(UNITS))
def test_method_function_and_functional_agree(name):
    call, _ = UNITS[name]
    tensor = mt.from_numpy(SAMPLE.copy())
    expected = call(tensor).numpy()
    args = FREE_ARGS.get(name, ())

    np.testing.assert_array_equal(getattr(mt, name)(tensor, *args).numpy(), expected)
    np.testing.assert_array_equal(
        getattr(mt.functional, name)(tensor, *args).numpy(), expected
    )


@pytest.mark.parametrize("name", sorted(UNITS))
def test_shape_and_dtype_are_preserved(name):
    call, _ = UNITS[name]
    for dtype in ("float32", "float64"):
        tensor = mt.Tensor(SAMPLE.reshape(13, 1).tolist(), dtype=dtype)
        out = call(tensor)
        assert out.shape == (13, 1)
        assert dtype in str(out.dtype)


def test_the_saturating_units_answer_where_the_definition_overflows():
    # exp(800) is inf in float64, so `log(sigmoid(x))` and
    # `x * tanh(log1p(exp(x)))` written out give -inf and NaN at these inputs.
    extreme = mt.from_numpy(np.array([-800.0, 800.0]))

    np.testing.assert_array_equal(extreme.logsigmoid().numpy(), [-800.0, 0.0])
    got = extreme.mish().numpy()
    assert got[0] == pytest.approx(0.0, abs=1e-300)
    assert got[1] == 800.0

    with np.errstate(over="ignore", divide="ignore"):
        assert np.log(_sigmoid(-800.0)) == -np.inf


def test_relu6_is_hardtanh_on_zero_to_six():
    tensor = mt.from_numpy(SAMPLE.copy())
    np.testing.assert_array_equal(
        tensor.relu6().numpy(), tensor.hardtanh(0.0, 6.0).numpy()
    )


def test_celu_has_the_continuous_slope_at_zero_that_elu_lacks():
    # `elu(alpha)` meets slope 1 at zero only when alpha is 1; `celu` rescales
    # the exponential so it does for every alpha. That is the whole difference.
    eps = 1e-6
    for alpha in (0.5, 1.0, 2.0):
        just_below = mt.Tensor([-eps], dtype="float64")
        just_above = mt.Tensor([eps], dtype="float64")
        left = (just_above.celu(alpha).item() - just_below.celu(alpha).item()) / (
            2 * eps
        )
        assert left == pytest.approx(1.0, abs=1e-5), alpha

    slope = (
        mt.Tensor([eps], dtype="float64").elu(2.0).item()
        - mt.Tensor([-eps], dtype="float64").elu(2.0).item()
    ) / (2 * eps)
    assert slope == pytest.approx(1.5, abs=1e-5)


def test_softshrink_is_continuous_where_hardshrink_jumps():
    at_the_edge = mt.Tensor([0.5 + 1e-9, 0.5 - 1e-9], dtype="float64")
    soft = at_the_edge.softshrink(0.5).numpy()
    hard = at_the_edge.hardshrink(0.5).numpy()

    assert abs(soft[0] - soft[1]) < 1e-8, "softshrink jumped at its threshold"
    assert abs(hard[0] - hard[1]) == pytest.approx(0.5, abs=1e-8)


def test_softmin_is_softmax_of_the_negation():
    values = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 5.0]])
    tensor = mt.from_numpy(values.copy())
    got = mt.softmin(tensor, dim=1).numpy()

    np.testing.assert_allclose(got.sum(axis=1), [1.0, 1.0], rtol=1e-15)
    np.testing.assert_allclose(
        got, mt.softmax(mt.from_numpy(-values), dim=1).numpy(), rtol=1e-15
    )
    # It ranks the smallest element highest, which is what distinguishes it.
    assert got[0, 0] > got[0, 2]


@pytest.mark.parametrize("name", sorted(UNITS))
def test_gradients_flow_and_the_flat_regions_do_not(name):
    call, _ = UNITS[name]
    # Deliberately includes the flat regions of every hard unit.
    values = np.array([-8.0, -1.0, 0.1, 1.0, 8.0])
    tensor = mt.Tensor(values, dtype="float64", requires_grad=True)
    call(tensor).sum().backward()
    grad = tensor.grad.numpy()

    assert np.all(np.isfinite(grad)), f"{name} produced a non-finite gradient"

    flat = {
        "relu6": [0, 1, 4],
        "hardtanh": [0, 4],
        "hardsigmoid": [0, 4],
        "hardswish": [0],
        "softshrink": [2],
        "threshold": [0, 1, 2],
    }.get(name, [])
    for index in flat:
        assert grad[index] == 0.0, f"{name} leaked a gradient at x={values[index]}"


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda t: t.hardtanh(1.0, -1.0), "min_val <= max_val"),
        (lambda t: t.softshrink(-0.5), "non-negative"),
        (lambda t: t.celu(0.0), "non-zero alpha"),
    ],
)
def test_invalid_parameters_are_rejected(call, message):
    with pytest.raises(ValueError, match=message):
        call(mt.from_numpy(np.array([1.0])))


@pytest.mark.parametrize("name", sorted(UNITS))
def test_integer_inputs_are_rejected(name):
    call, _ = UNITS[name]
    with pytest.raises(ValueError):
        call(mt.Tensor.arange(-3, 4, dtype="int64"))


@pytest.mark.parametrize("name", sorted(UNITS))
def test_empty_tensors_come_back_empty(name):
    call, _ = UNITS[name]
    assert call(mt.from_numpy(np.array([]))).shape == (0,)
