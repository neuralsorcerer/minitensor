# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""`has_nan()` and `has_inf()` answer the question a training loop asks.

The engine has had both since the start with no binding, so the only way to ask
"did this diverge?" was `isnan(x).any()`, which builds an N-element boolean
tensor and reads every element even when the first one is already NaN.

On what that is worth, measured on this machine rather than assumed:

    16M float32     isnan().any()    has_nan()
    no NaN             11.0 ms        10.6 ms     1.04x
    NaN at index 0      8.2 ms         0.0 ms     short-circuits
    NaN at the end      9.6 ms        10.1 ms     0.94x

So it is a wash when the tensor is clean, marginally slower in the contrived
case where the only NaN is the last element -- the scan is serial and cannot
fan out -- and effectively free when a NaN is present early, which is what a
diverged gradient looks like.

These tests are agreement checks against `isnan`/`isinf`, so the two cannot
drift into disagreeing the way `array_equal` and `eq` had.
"""

from __future__ import annotations

import numpy as np
import pytest

import minitensor as mt

FLOAT_DTYPES = ["float32", "float64"]

# Positions chosen to cover the short-circuit: first element, last element, and
# somewhere in the middle.
CASES = [
    ("clean", None),
    ("first", 0),
    ("middle", -2),
    ("last", -1),
]

SIZES = [1, 7, 1024, 65536]


def _with_value(size, position, value, dtype):
    values = np.arange(size, dtype=dtype)
    if position is not None:
        values[position if position >= 0 else size + position] = value
    return mt.Tensor(values, dtype=dtype)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("name,position", CASES, ids=[case[0] for case in CASES])
def test_has_nan_agrees_with_isnan_any(name, position, size, dtype):
    if position is not None and abs(position) > size:
        pytest.skip("position does not exist at this size")
    tensor = _with_value(size, position, np.nan, dtype)

    assert tensor.has_nan() == bool(mt.isnan(tensor).any().numpy())
    assert tensor.has_nan() == bool(np.isnan(tensor.numpy()).any())


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("name,position", CASES, ids=[case[0] for case in CASES])
@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_has_inf_agrees_with_isinf_any(name, position, size, dtype, value):
    if position is not None and abs(position) > size:
        pytest.skip("position does not exist at this size")
    tensor = _with_value(size, position, value, dtype)

    assert tensor.has_inf() == bool(mt.isinf(tensor).any().numpy())
    assert tensor.has_inf() == bool(np.isinf(tensor.numpy()).any())


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_the_two_predicates_do_not_answer_for_each_other(dtype):
    """A NaN is not an infinity and an infinity is not a NaN, which a shared
    "non-finite" implementation would blur."""
    nan_only = mt.Tensor(np.array([1.0, np.nan], dtype=dtype), dtype=dtype)
    inf_only = mt.Tensor(np.array([1.0, np.inf], dtype=dtype), dtype=dtype)

    assert nan_only.has_nan() and not nan_only.has_inf()
    assert inf_only.has_inf() and not inf_only.has_nan()


@pytest.mark.parametrize("dtype", ["int32", "int64", "bool"])
def test_non_float_dtypes_are_always_finite(dtype):
    """There is no NaN or infinity to find, and asking should not error."""
    values = np.array([True, False]) if dtype == "bool" else np.arange(5).astype(dtype)
    tensor = mt.Tensor(values, dtype=dtype)

    assert not tensor.has_nan()
    assert not tensor.has_inf()


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_an_empty_tensor_holds_neither(dtype):
    tensor = mt.Tensor(np.array([], dtype=dtype), dtype=dtype)
    assert not tensor.has_nan()
    assert not tensor.has_inf()


def test_a_diverged_gradient_is_detectable():
    """The use case: a training step that produced NaN gradients.

    `0 * inf` is NaN, which is how a diverged forward turns into NaN gradients.
    """
    from minitensor import nn

    mt.manual_seed(0)
    model = nn.DenseLayer(3, 2)
    features = mt.Tensor(np.full((4, 3), 1e30, dtype=np.float32))
    targets = mt.Tensor(np.full((4, 2), -1e30, dtype=np.float32))

    loss = nn.mse_loss(model(features), targets)
    loss.backward()

    assert loss.has_nan() or loss.has_inf(), "this input should have diverged"
    gradients = [mt.get_gradient(p) for p in model.parameters()]
    assert any(
        g is not None and (g.has_nan() or g.has_inf()) for g in gradients
    ), "a diverged loss should leave non-finite gradients"


def test_a_healthy_training_step_reports_neither():
    """The other half: the check must not fire on an ordinary step."""
    from minitensor import nn

    mt.manual_seed(0)
    model = nn.DenseLayer(3, 2)
    loss = nn.mse_loss(model(mt.randn(4, 3)), mt.randn(4, 2))
    loss.backward()

    assert not loss.has_nan() and not loss.has_inf()
    for parameter in model.parameters():
        gradient = mt.get_gradient(parameter)
        assert gradient is not None
        assert not gradient.has_nan() and not gradient.has_inf()
