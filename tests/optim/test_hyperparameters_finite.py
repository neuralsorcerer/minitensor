# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Optimizer hyperparameters are refused unless they are finite numbers.

Each check was a comparison -- `lr <= 0`, `eps <= 0`, `weight_decay < 0` --
and NaN fails every comparison, so a NaN passed all of them and the first
step turned every parameter into NaN. Infinity passed the lower bounds too.
"""

from __future__ import annotations

import math

import pytest

import minitensor as mt

O = mt.optim


def _param():
    return mt.tensor([1.0, -2.0], dtype="float64", requires_grad=True)


CASES = [
    (O.SGD, "lr", {}),
    (O.SGD, "momentum", {"lr": 0.1}),
    (O.SGD, "weight_decay", {"lr": 0.1}),
    (O.Adam, "lr", {}),
    (O.Adam, "eps", {}),
    (O.Adam, "weight_decay", {}),
    (O.AdamW, "weight_decay", {}),
    (O.RMSprop, "lr", {}),
    (O.RMSprop, "momentum", {"lr": 0.1}),
    (O.Adagrad, "lr", {}),
    (O.Lion, "lr", {}),
]


@pytest.mark.parametrize("value", [math.nan, math.inf], ids=["nan", "inf"])
@pytest.mark.parametrize(
    ("cls", "name", "base"), CASES, ids=[f"{c.__name__}-{n}" for c, n, _ in CASES]
)
def test_a_non_finite_hyperparameter_is_refused(cls, name, base, value):
    kwargs = dict(base)
    kwargs[name] = value
    if cls is O.RMSprop and "lr" not in kwargs:
        kwargs["lr"] = 0.1
    with pytest.raises(ValueError, match="finite"):
        cls([_param()], **kwargs)


@pytest.mark.parametrize("value", [1.5, -0.1, math.nan])
def test_sgd_dampening_must_lie_in_the_unit_interval(value):
    with pytest.raises(ValueError, match=r"Dampening must be in the range \[0, 1\]"):
        O.SGD([_param()], lr=0.1, momentum=0.9, dampening=value)


def test_the_learning_rate_setter_takes_zero_but_not_negative_or_nan():
    optimizer = O.SGD([_param()], lr=0.1)
    optimizer.lr = 0.0
    assert optimizer.lr == 0.0
    for value in (-1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="non-negative and finite"):
            optimizer.lr = value
    assert optimizer.lr == 0.0
